//===--- cuda/src/cuda_compat.h -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CUDA compatibility layer enabling us to compile using new APIs.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_CUDA_CUDACOMPAT_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_CUDA_CUDACOMPAT_H

#include "APIHelpers.h"

#include <cuda.h>

API_HELPER_OPTIONAL(CUresult, cuMemPrefetchBatchAsync, CUdeviceptr *dptrs,
                    size_t *sizes, size_t count, CUmemLocation *prefetchLocs,
                    size_t *prefetchLocIdxs, size_t numPrefetchLocs,
                    unsigned long long flags, CUstream hStream)

#endif
