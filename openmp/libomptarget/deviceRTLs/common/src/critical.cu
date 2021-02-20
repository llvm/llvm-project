//===------ critical.cu - NVPTX OpenMP critical ------------------ CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of critical with KMPC interface
//
//===----------------------------------------------------------------------===//
#pragma omp declare target

#include "common/debug.h"
#include "interface.h"

EXTERN
void __kmpc_critical(kmp_Ident *loc, int32_t global_tid,
                     kmp_CriticalName *lck) {
  PRINT0(LD_IO, "call to kmpc_critical()\n");
  omp_set_lock((omp_lock_t *)lck);
}

EXTERN
void __kmpc_end_critical(kmp_Ident *loc, int32_t global_tid,
                         kmp_CriticalName *lck) {
  PRINT0(LD_IO, "call to kmpc_end_critical()\n");
  omp_unset_lock((omp_lock_t *)lck);
}

#pragma omp end declare target
