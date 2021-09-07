//===--- amdgpu/impl/data.cpp ------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
namespace Runtime {
hsa_status_t HostMalloc(void **ptr, size_t size,
                        hsa_amd_memory_pool_t MemoryPool) {
  hsa_status_t err = hsa_amd_memory_pool_allocate(MemoryPool, size, 0, ptr);
  DP("Malloced %p\n", *ptr);
  if (err == HSA_STATUS_SUCCESS) {
    err = core::allow_access_to_all_gpu_agents(*ptr);
  }
  return err;
}

hsa_status_t Memfree(void *ptr) {
  hsa_status_t err = hsa_amd_memory_pool_free(ptr);
  DP("Freed %p\n", ptr);
  return err;
}
} // namespace Runtime
} // namespace core
