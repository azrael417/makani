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
DM = None

def get_size(name: str) -> int:
    return DM.get_group_size(name)

    
def get_rank(name: str) -> int:
    return DM.get_group_rank(name)

    
def get_group(name: str):
    return DM.group(name)

# TODO: we need this
#def get_root(comm_id: Union[str, int]) -> int:
#    DM.

# specialized routines for world comms
def get_world_size():
    return DM.world_size

    
def get_world_rank():
    return DM.rank


def get_local_rank():
    return DM.local_rank


def get_names():
    return DM.group_names


def is_distributed(name: str):
    return (name in DM.group_names)


# get 
def init(params, verbose=False):

    # call basic init first
    DistributedManager.initialize()
    
    # extract manager object
    global DM
    DM = DistributedManager()

    # do individual wireup for model parallel comms:
    model_parallel_sizes = params.get("model_parallel_sizes", [1, 1, 1, 1])
    model_parallel_names = params.get("model_parallel_names", ["h", "w", "fin", "fout"])

    # create process group config:
    world = ProcessGroupNode('world')
    pconfig = ProcessGroupConfig(world)

    # add leaf nodes:
    # data
    pconfig.add_node(world, "data")
    # model
    pconfig.add_node(world, "model")
    # spatial and matmul
    pconfig.add_node(pconfig.get_node("model"), "spatial")
    pconfig.add_node(pconfig.get_node("model"), "matmul")
    # subgroups for spatial
    pconfig.add_node(pconfig.get_node("spatial"), "h")
    pconfig.add_node(pconfig.get_node("spatial"), "w")
    # subgroups for matmul:
    pconfig.add_node(pconfig.get_node("matmul"), "fin")
    pconfig.add_node(pconfig.get_node("matmul"), "fout")

    # set up leaf sizes
    leaf_config = {k: v for k,v in zip(model_parallel_names, model_parallel_sizes)}
    pconfig.set_leaf_group_sizes(leaf_config, update_parent_sizes=True)

    # create remaining process groups
    DM.create_groups_from_config(pconfig, verbose=verbose)
    
    return DM.group_size("model")
