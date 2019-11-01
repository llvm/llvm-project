//===------ unity.cu - Unity build of NVPTX deviceRTL ------------ CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support compilers, specifically NVCC, which have not implemented link time
// optimisation. This removes the runtime cost of moving inline functions into
// source files in exchange for preventing efficient incremental builds.
//
//===----------------------------------------------------------------------===//

#include "src/cancel.cu"
#include "src/critical.cu"
#include "src/data_sharing.cu"
#include "src/libcall.cu"
#include "src/loop.cu"
#include "src/omp_data.cu"
#include "src/omptarget-nvptx.cu"
#include "src/parallel.cu"
#include "src/reduction.cu"
#include "src/support.cu"
#include "src/sync.cu"
#include "src/task.cu"
