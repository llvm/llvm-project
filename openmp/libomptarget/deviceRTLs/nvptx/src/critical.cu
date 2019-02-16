//===------ critical.cu - NVPTX OpenMP critical ------------------ CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of critical with KMPC interface
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#include "omptarget-nvptx.h"

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
