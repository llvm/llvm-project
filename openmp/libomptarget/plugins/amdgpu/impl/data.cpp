//===--- amdgpu/impl/data.cpp ------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
// Notified per clause 4(b) of the license.
//
//===----------------------------------------------------------------------===//
#include "impl_runtime.h"
#include "hsa_api.h"
#include "internal.h"
#include "rt.h"
#include <cassert>
#include <stdio.h>
#include <string.h>
#include <vector>

using core::TaskImpl;

namespace core {
extern "C" __attribute__((weak)) void
fort_ptr_assign_i8(void *arg0, void *arg1, void *arg2, void *arg3, void *arg4);
hsa_status_t Runtime::FtnAssignWrapper(void *arg0, void *arg1, void *arg2, void *arg3, void *arg4) {
  fort_ptr_assign_i8(arg0, arg1, arg2, arg3, arg4);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t Runtime::HostMalloc(void **ptr, size_t size,
                        hsa_amd_memory_pool_t MemoryPool) {
  hsa_status_t err = hsa_amd_memory_pool_allocate(MemoryPool, size, 0, ptr);
  DP("Malloced %p\n", *ptr);
  if (err == HSA_STATUS_SUCCESS) {
    err = core::allow_access_to_all_gpu_agents(*ptr);
  }
  return err;
}

hsa_status_t Runtime::Memfree(void *ptr) {
  hsa_status_t err = hsa_amd_memory_pool_free(ptr);
  DP("Freed %p\n", ptr);
  return err;
}

} // namespace core
