# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import logging
from utils.logging_utils import disable_logging
import math
from typing import Union
import numpy as np

# we are using the distributed manager from modulus
from modulus.distributed.manager import DistributedManager
from modulus.distributed.config import ProcessGroupNode, ProcessGroupConfig

# we need this
_DM = None
_COMM_ROOTS = {}

def get_size(name: str) -> int:
    return _DM.group_size(name)

    
def get_rank(name: str) -> int:
    return _DM.group_rank(name)

    
def get_group(name: str):
    return _DM.group(name)


def get_root(name: str) -> int:
    
    if name in _COMM_ROOTS:
        return _COMM_ROOTS[name]
    else:
        return 0

    
# specialized routines for world comms
def get_world_size():
    return _DM.world_size

    
def get_world_rank():
    return _DM.rank


def get_local_rank():
    return _DM.local_rank


def get_names():
    return [name for name in _DM.group_names if (not name.startswith("__orthogonal_to"))]


def is_distributed(name: str):
    return (name in _DM.group_names)


# get 
def init(params, verbose=False):

    # call basic init first
    DistributedManager.initialize()
    
    # extract manager object
    global _DM
    _DM = DistributedManager()

    # do individual wireup for model parallel comms:
    model_parallel_sizes = params.get("model_parallel_sizes", [1, 1, 1, 1])
    model_parallel_names = params.get("model_parallel_names", ["h", "w", "fin", "fout"])

    # create process group config:
    world = ProcessGroupNode('world', size=_DM.world_size)
    pconfig = ProcessGroupConfig(world)

    # add nodes:
    # model
    pconfig.add_node(ProcessGroupNode("model"), parent="world")
    # spatial and matmul
    pconfig.add_node(ProcessGroupNode("spatial"), parent="model")
    pconfig.add_node(ProcessGroupNode("matmul"), parent="model")
    # subgroups for spatial
    pconfig.add_node(ProcessGroupNode("h"), parent="spatial")
    pconfig.add_node(ProcessGroupNode("w"), parent="spatial")
    # subgroups for matmul:
    pconfig.add_node(ProcessGroupNode("fin"), parent="matmul")
    pconfig.add_node(ProcessGroupNode("fout"), parent="matmul")
    # add data node last
    pconfig.add_node(ProcessGroupNode("data"), parent="world")

    # set up leaf sizes
    leaf_config = {k: v for k,v in zip(model_parallel_names, model_parallel_sizes)}
    leaf_config["data"] = _DM.world_size // math.prod(leaf_config.values())
    pconfig.set_leaf_group_sizes(leaf_config, update_parent_sizes=True)

    # create remaining process groups
    _DM.create_groups_from_config(pconfig, verbose=(verbose and (_DM.rank == 0)))

    # get comm roots:
    global _COMM_ROOTS
    for gname in get_names():
        rank = _DM.rank
        for grp in _DM._group_ranks[gname]:
            if rank in grp:
                _COMM_ROOTS[gname] = min(grp)

    # print debug
    #import torch
    #for rank in range(_DM.world_size):
    #    if rank == _DM.rank:
    #        print(f"{rank}: groups:")
    #        for gname in get_names():
    #            print(f"\t{gname}: {_DM._group_ranks[gname]}, root={_COMM_ROOTS[gname]}")
    #    torch.distributed.barrier(device_ids=[_DM.local_rank])
    
    return _DM.group_size("model")
