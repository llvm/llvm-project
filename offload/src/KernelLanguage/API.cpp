//===------ API.cpp - Kernel Language (CUDA/HIP) entry points ----- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Shared/APITypes.h"

#include <cstdio>

struct dim3 {
  unsigned x = 0, y = 0, z = 0;
};

struct __omp_kernel_t {
  dim3 __grid_size;
  dim3 __block_size;
  size_t __shared_memory;

  void *__stream;
};

static __omp_kernel_t __current_kernel = {};
#pragma omp threadprivate(__current_kernel);

extern "C" {

// TODO: There is little reason we need to keep these names or the way calls are
// issued. For now we do to avoid modifying Clang's CUDA codegen. Unclear when
// we actually need to push/pop configurations.
unsigned __llvmPushCallConfiguration(dim3 __grid_size, dim3 __block_size,
                                     size_t __shared_memory, void *__stream) {
  __omp_kernel_t &__kernel = __current_kernel;
  __kernel.__grid_size = __grid_size;
  __kernel.__block_size = __block_size;
  __kernel.__shared_memory = __shared_memory;
  __kernel.__stream = __stream;
  return 0;
}

unsigned __llvmPopCallConfiguration(dim3 *__grid_size, dim3 *__block_size,
                                    size_t *__shared_memory, void *__stream) {
  __omp_kernel_t &__kernel = __current_kernel;
  *__grid_size = __kernel.__grid_size;
  *__block_size = __kernel.__block_size;
  *__shared_memory = __kernel.__shared_memory;
  *((void **)__stream) = __kernel.__stream;
  return 0;
}

int __tgt_target_kernel(void *Loc, int64_t DeviceId, int32_t NumTeams,
                        int32_t ThreadLimit, const void *HostPtr,
                        KernelArgsTy *Args);

unsigned llvmLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                          void *args, size_t sharedMem, void *stream) {
  KernelArgsTy Args = {};
  Args.DynCGroupMem = sharedMem;
  Args.NumTeams[0] = gridDim.x;
  Args.NumTeams[1] = gridDim.y;
  Args.NumTeams[2] = gridDim.z;
  Args.ThreadLimit[0] = blockDim.x;
  Args.ThreadLimit[1] = blockDim.y;
  Args.ThreadLimit[2] = blockDim.z;
  Args.ArgPtrs = reinterpret_cast<void **>(args);
  Args.Flags.IsCUDA = true;
  return __tgt_target_kernel(nullptr, 0, gridDim.x, blockDim.x, func, &Args);
}
}
