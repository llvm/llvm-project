//===--- amdgcn_interface.h - OpenMP interface definitions ------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _AMDGCN_INTERFACE_H_
#define _AMDGCN_INTERFACE_H_

#include <stddef.h>
#include <stdint.h>

#define EXTERN extern "C"
typedef uint32_t omp_lock_t; /* arbitrary type of the right length */

////////////////////////////////////////////////////////////////////////////////
// OpenMP interface
////////////////////////////////////////////////////////////////////////////////

EXTERN uint32_t __kmpc_amdgcn_gpu_num_threads(void);

EXTERN int omp_get_device_num(void);
EXTERN int omp_ext_get_warp_id(void);
EXTERN int omp_ext_get_lane_id(void);
EXTERN int omp_ext_get_master_thread_id(void);
EXTERN int omp_ext_get_smid(void);
EXTERN int omp_ext_is_spmd_mode(void);
EXTERN unsigned long long omp_ext_get_active_threads_mask(void);

////////////////////////////////////////////////////////////////////////////////
// kmp specifc types
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// external interface
////////////////////////////////////////////////////////////////////////////////

typedef struct ident ident_t;
typedef ident_t kmp_Ident;

EXTERN uint32_t __kmpc_amdgcn_gpu_num_threads();

#endif
