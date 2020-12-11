//===--- amdgcn_intrinsics.h - Intrinsics used by deviceRTL ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _AMDGCN_INTRINSICS_H_
#define _AMDGCN_INTRINSICS_H_

#ifndef EXTERN
#error "Expected definition of EXTERN"
#endif

#include <stdint.h>

#ifdef _OPENMP
// Openmp doesn't pull these builtins into scope, but does error if the type is
// incorrect This may be a quirk of openmp's compile for host + device
// assumption, where these don't resolve to anything on the host

EXTERN uint32_t __builtin_amdgcn_atomic_inc32(volatile uint32_t *, uint32_t,
                                              uint32_t, const char *);
EXTERN void __builtin_amdgcn_s_barrier(void);
EXTERN void __builtin_amdgcn_fence(uint32_t, const char *);

EXTERN void __builtin_amdgcn_s_sleep(int);

EXTERN uint32_t __builtin_amdgcn_workitem_id_x(void);
EXTERN uint32_t __builtin_amdgcn_workgroup_id_x(void);
EXTERN uint16_t __builtin_amdgcn_workgroup_size_x(void);
EXTERN uint32_t __builtin_amdgcn_grid_size_x(void);

EXTERN uint64_t __builtin_amdgcn_s_memrealtime(void);
EXTERN uint32_t __builtin_amdgcn_s_getreg(int32_t);
EXTERN uint64_t __builtin_amdgcn_read_exec(void);

EXTERN __attribute__((address_space(4))) void *
__builtin_amdgcn_dispatch_ptr() noexcept;

EXTERN uint32_t __builtin_amdgcn_mbcnt_lo(uint32_t, uint32_t);
EXTERN uint32_t __builtin_amdgcn_mbcnt_hi(uint32_t, uint32_t);
EXTERN int32_t __builtin_amdgcn_ds_bpermute(int32_t, int32_t);
#endif

#endif
