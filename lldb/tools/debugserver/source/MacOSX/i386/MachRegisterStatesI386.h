//===-- MachRegisterStatesI386.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Created by Sean Callanan on 3/16/11.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_I386_MACHREGISTERSTATESI386_H
#define LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_I386_MACHREGISTERSTATESI386_H

#include <inttypes.h>

#define __i386_THREAD_STATE 1
#define __i386_FLOAT_STATE 2
#define __i386_EXCEPTION_STATE 3
#define __i386_DEBUG_STATE 10
#define __i386_AVX_STATE 16
#define __i386_AVX512F_STATE 19

typedef struct {
  uint32_t __eax;
  uint32_t __ebx;
  uint32_t __ecx;
  uint32_t __edx;
  uint32_t __edi;
  uint32_t __esi;
  uint32_t __ebp;
  uint32_t __esp;
  uint32_t __ss;
  uint32_t __eflags;
  uint32_t __eip;
  uint32_t __cs;
  uint32_t __ds;
  uint32_t __es;
  uint32_t __fs;
  uint32_t __gs;
} __i386_thread_state_t;

typedef struct {
  uint16_t __invalid : 1;
  uint16_t __denorm : 1;
  uint16_t __zdiv : 1;
  uint16_t __ovrfl : 1;
  uint16_t __undfl : 1;
  uint16_t __precis : 1;
  uint16_t __PAD1 : 2;
  uint16_t __pc : 2;
  uint16_t __rc : 2;
  uint16_t __PAD2 : 1;
  uint16_t __PAD3 : 3;
} __i386_fp_control_t;

typedef struct {
  uint16_t __invalid : 1;
  uint16_t __denorm : 1;
  uint16_t __zdiv : 1;
  uint16_t __ovrfl : 1;
  uint16_t __undfl : 1;
  uint16_t __precis : 1;
  uint16_t __stkflt : 1;
  uint16_t __errsumm : 1;
  uint16_t __c0 : 1;
  uint16_t __c1 : 1;
  uint16_t __c2 : 1;
  uint16_t __tos : 3;
  uint16_t __c3 : 1;
  uint16_t __busy : 1;
} __i386_fp_status_t;

typedef struct {
  uint8_t __mmst_reg[10];
  uint8_t __mmst_rsrv[6];
} __i386_mmst_reg;

typedef struct { uint8_t __xmm_reg[16]; } __i386_xmm_reg;

typedef struct {
  uint32_t __fpu_reserved[2];
  __i386_fp_control_t __fpu_fcw;
  __i386_fp_status_t __fpu_fsw;
  uint8_t __fpu_ftw;
  uint8_t __fpu_rsrv1;
  uint16_t __fpu_fop;
  uint32_t __fpu_ip;
  uint16_t __fpu_cs;
  uint16_t __fpu_rsrv2;
  uint32_t __fpu_dp;
  uint16_t __fpu_ds;
  uint16_t __fpu_rsrv3;
  uint32_t __fpu_mxcsr;
  uint32_t __fpu_mxcsrmask;
  __i386_mmst_reg __fpu_stmm0;
  __i386_mmst_reg __fpu_stmm1;
  __i386_mmst_reg __fpu_stmm2;
  __i386_mmst_reg __fpu_stmm3;
  __i386_mmst_reg __fpu_stmm4;
  __i386_mmst_reg __fpu_stmm5;
  __i386_mmst_reg __fpu_stmm6;
  __i386_mmst_reg __fpu_stmm7;
  __i386_xmm_reg __fpu_xmm0;
  __i386_xmm_reg __fpu_xmm1;
  __i386_xmm_reg __fpu_xmm2;
  __i386_xmm_reg __fpu_xmm3;
  __i386_xmm_reg __fpu_xmm4;
  __i386_xmm_reg __fpu_xmm5;
  __i386_xmm_reg __fpu_xmm6;
  __i386_xmm_reg __fpu_xmm7;
  uint8_t __fpu_rsrv4[14 * 16];
  uint32_t __fpu_reserved1;
} __i386_float_state_t;

typedef struct {
  uint32_t __fpu_reserved[2];
  __i386_fp_control_t __fpu_fcw;
  __i386_fp_status_t __fpu_fsw;
  uint8_t __fpu_ftw;
  uint8_t __fpu_rsrv1;
  uint16_t __fpu_fop;
  uint32_t __fpu_ip;
  uint16_t __fpu_cs;
  uint16_t __fpu_rsrv2;
  uint32_t __fpu_dp;
  uint16_t __fpu_ds;
  uint16_t __fpu_rsrv3;
  uint32_t __fpu_mxcsr;
  uint32_t __fpu_mxcsrmask;
  __i386_mmst_reg __fpu_stmm0;
  __i386_mmst_reg __fpu_stmm1;
  __i386_mmst_reg __fpu_stmm2;
  __i386_mmst_reg __fpu_stmm3;
  __i386_mmst_reg __fpu_stmm4;
  __i386_mmst_reg __fpu_stmm5;
  __i386_mmst_reg __fpu_stmm6;
  __i386_mmst_reg __fpu_stmm7;
  __i386_xmm_reg __fpu_xmm0;
  __i386_xmm_reg __fpu_xmm1;
  __i386_xmm_reg __fpu_xmm2;
  __i386_xmm_reg __fpu_xmm3;
  __i386_xmm_reg __fpu_xmm4;
  __i386_xmm_reg __fpu_xmm5;
  __i386_xmm_reg __fpu_xmm6;
  __i386_xmm_reg __fpu_xmm7;
  uint8_t __fpu_rsrv4[14 * 16];
  uint32_t __fpu_reserved1;
  uint8_t __avx_reserved1[64];
  __i386_xmm_reg __fpu_ymmh0;
  __i386_xmm_reg __fpu_ymmh1;
  __i386_xmm_reg __fpu_ymmh2;
  __i386_xmm_reg __fpu_ymmh3;
  __i386_xmm_reg __fpu_ymmh4;
  __i386_xmm_reg __fpu_ymmh5;
  __i386_xmm_reg __fpu_ymmh6;
  __i386_xmm_reg __fpu_ymmh7;
} __i386_avx_state_t;

typedef struct { uint8_t __ymm_reg[32]; } __i386_ymm_reg;
typedef struct { uint8_t __opmask_reg[8]; } __i386_opmask_reg;

typedef struct {
  uint32_t __fpu_reserved[2];
  __i386_fp_control_t __fpu_fcw;
  __i386_fp_status_t __fpu_fsw;
  uint8_t __fpu_ftw;
  uint8_t __fpu_rsrv1;
  uint16_t __fpu_fop;
  uint32_t __fpu_ip;
  uint16_t __fpu_cs;
  uint16_t __fpu_rsrv2;
  uint32_t __fpu_dp;
  uint16_t __fpu_ds;
  uint16_t __fpu_rsrv3;
  uint32_t __fpu_mxcsr;
  uint32_t __fpu_mxcsrmask;
  __i386_mmst_reg __fpu_stmm0;
  __i386_mmst_reg __fpu_stmm1;
  __i386_mmst_reg __fpu_stmm2;
  __i386_mmst_reg __fpu_stmm3;
  __i386_mmst_reg __fpu_stmm4;
  __i386_mmst_reg __fpu_stmm5;
  __i386_mmst_reg __fpu_stmm6;
  __i386_mmst_reg __fpu_stmm7;
  __i386_xmm_reg __fpu_xmm0;
  __i386_xmm_reg __fpu_xmm1;
  __i386_xmm_reg __fpu_xmm2;
  __i386_xmm_reg __fpu_xmm3;
  __i386_xmm_reg __fpu_xmm4;
  __i386_xmm_reg __fpu_xmm5;
  __i386_xmm_reg __fpu_xmm6;
  __i386_xmm_reg __fpu_xmm7;
  uint8_t __fpu_rsrv4[14 * 16];
  uint32_t __fpu_reserved1;
  uint8_t __avx_reserved1[64];
  __i386_xmm_reg __fpu_ymmh0;
  __i386_xmm_reg __fpu_ymmh1;
  __i386_xmm_reg __fpu_ymmh2;
  __i386_xmm_reg __fpu_ymmh3;
  __i386_xmm_reg __fpu_ymmh4;
  __i386_xmm_reg __fpu_ymmh5;
  __i386_xmm_reg __fpu_ymmh6;
  __i386_xmm_reg __fpu_ymmh7;
  __i386_opmask_reg __fpu_k0;
  __i386_opmask_reg __fpu_k1;
  __i386_opmask_reg __fpu_k2;
  __i386_opmask_reg __fpu_k3;
  __i386_opmask_reg __fpu_k4;
  __i386_opmask_reg __fpu_k5;
  __i386_opmask_reg __fpu_k6;
  __i386_opmask_reg __fpu_k7;
  __i386_ymm_reg __fpu_zmmh0;
  __i386_ymm_reg __fpu_zmmh1;
  __i386_ymm_reg __fpu_zmmh2;
  __i386_ymm_reg __fpu_zmmh3;
  __i386_ymm_reg __fpu_zmmh4;
  __i386_ymm_reg __fpu_zmmh5;
  __i386_ymm_reg __fpu_zmmh6;
  __i386_ymm_reg __fpu_zmmh7;
} __i386_avx512f_state_t;

typedef struct {
  uint32_t __trapno;
  uint32_t __err;
  uint32_t __faultvaddr;
} __i386_exception_state_t;

typedef struct {
  uint32_t __dr0;
  uint32_t __dr1;
  uint32_t __dr2;
  uint32_t __dr3;
  uint32_t __dr4;
  uint32_t __dr5;
  uint32_t __dr6;
  uint32_t __dr7;
} __i386_debug_state_t;

#endif
