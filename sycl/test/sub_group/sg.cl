//==-------------- sg.cl - OpenCL reference kernel file --------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
struct Data {
  uint local_id;
  uint local_range;
  uint max_local_range;
  uint group_id;
  uint group_range;
  uint uniform_group_range;
};
__kernel void ocl_subgr(__global struct Data *a) {
  uint id = get_global_id(0);
  a[id].local_id = get_sub_group_local_id();
  a[id].local_range = get_sub_group_size();
  a[id].max_local_range = get_max_sub_group_size();
  a[id].group_id = get_sub_group_id();
  a[id].group_range = get_num_sub_groups();
  a[id].uniform_group_range = get_num_sub_groups();
}
