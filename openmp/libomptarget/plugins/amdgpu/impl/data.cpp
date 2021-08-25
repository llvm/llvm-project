//===--- amdgpu/impl/data.cpp ------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "atmi_runtime.h"
#include "hsa_api.h"
#include "internal.h"
#include "rt.h"
#include <cassert>
#include <stdio.h>
#include <string.h>
#include <vector>

using core::TaskImpl;

namespace core {

hsa_status_t Runtime::HostMalloc(void **ptr, size_t size,
                                 hsa_amd_memory_pool_t MemoryPool) {
  hsa_status_t err = hsa_amd_memory_pool_allocate(MemoryPool, size, 0, ptr);
  DEBUG_PRINT("Malloced %p\n", *ptr);

  if (err == HSA_STATUS_SUCCESS) {
    err = core::allow_access_to_all_gpu_agents(*ptr);
  }
  return (err == HSA_STATUS_SUCCESS) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t Runtime::Memfree(void *ptr) {
  hsa_status_t err = hsa_amd_memory_pool_free(ptr);
  DEBUG_PRINT("Freed %p\n", ptr);

  return (err == HSA_STATUS_SUCCESS) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

} // namespace core
