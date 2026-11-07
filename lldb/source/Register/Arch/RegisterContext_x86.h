//===-- RegisterContext_x86.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_REGISTER_ARCH_REGISTERCONTEXT_X86_H
#define LLDB_SOURCE_REGISTER_ARCH_REGISTERCONTEXT_X86_H

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/Compiler.h"

namespace lldb_private {
// Generic floating-point registers

LLVM_PACKED_START
struct MMSRegComp {
  uint64_t mantissa;
  uint16_t sign_exp;
};

struct MMSReg {
  union {
    uint8_t bytes[10];
    MMSRegComp comp;
  };
  uint8_t pad[6];
};
LLVM_PACKED_END

static_assert(sizeof(MMSRegComp) == 10, "MMSRegComp is not 10 bytes of size");
static_assert(sizeof(MMSReg) == 16, "MMSReg is not 16 bytes of size");

struct XMMReg {
  uint8_t bytes[16]; // 128-bits for each XMM register
};

// i387_fxsave_struct
struct FXSAVE {
  uint16_t fctrl;     // FPU Control Word (fcw)
  uint16_t fstat;     // FPU Status Word (fsw)
  uint16_t ftag;      // FPU Tag Word (ftw)
  uint16_t fop;       // Last Instruction Opcode (fop)
  union {
    struct {
      uint64_t fip; // Instruction Pointer
      uint64_t fdp; // Data Pointer
    } x86_64;
    struct {
      uint32_t fioff; // FPU IP Offset (fip)
      uint32_t fiseg; // FPU IP Selector (fcs)
      uint32_t fooff; // FPU Operand Pointer Offset (foo)
      uint32_t foseg; // FPU Operand Pointer Selector (fos)
    } i386_; // Added _ in the end to avoid error with gcc defining i386 in some
             // cases
  } ptr;
  uint32_t mxcsr;     // MXCSR Register State
  uint32_t mxcsrmask; // MXCSR Mask
  MMSReg stmm[8];     // 8*16 bytes for each FP-reg = 128 bytes
  XMMReg xmm[16];     // 16*16 bytes for each XMM-reg = 256 bytes
  uint8_t padding1[48];
  uint64_t xcr0;
  uint8_t padding2[40];
};

// Extended floating-point registers

struct YMMHReg {
  uint8_t bytes[16]; // 16 * 8 bits for the high bytes of each YMM register
};

struct YMMReg {
  uint8_t bytes[32]; // 16 * 16 bits for each YMM register
};

struct YMM {
  YMMReg ymm[16]; // assembled from ymmh and xmm registers
};

struct MPXReg {
  uint8_t bytes[16]; // MPX 128 bit bound registers
};

struct MPXCsr {
  uint8_t bytes[8]; // MPX 64 bit bndcfgu and bndstatus registers (collectively
                    // BNDCSR state)
};

struct MPX {
  MPXReg mpxr[4];
  MPXCsr mpxc[2];
};

LLVM_PACKED_START
struct XSAVE_HDR {
  enum class XFeature : uint64_t {
    FP = 1,
    SSE = FP << 1,
    YMM = SSE << 1,
    BNDREGS = YMM << 1,
    BNDCSR = BNDREGS << 1,
    OPMASK = BNDCSR << 1,
    ZMM_Hi256 = OPMASK << 1,
    Hi16_ZMM = ZMM_Hi256 << 1,
    PT = Hi16_ZMM << 1,
    PKRU = PT << 1,
    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue*/ PKRU)
  };

  XFeature xstate_bv; // OS enabled xstate mask to determine the extended states
                      // supported by the processor
  XFeature xcomp_bv;  // Mask to indicate the format of the XSAVE area and of
                      // the XRSTOR instruction
  uint64_t reserved1[1];
  uint64_t reserved2[5];
};
static_assert(sizeof(XSAVE_HDR) == 64, "XSAVE_HDR layout incorrect");
LLVM_PACKED_END

// x86 extensions to FXSAVE (i.e. for AVX and MPX processors)
LLVM_PACKED_START
struct XSAVE {
  FXSAVE i387;      // floating point registers typical in i387_fxsave_struct
  XSAVE_HDR header; // The xsave_hdr_struct can be used to determine if the
                    // following extensions are usable
  YMMHReg ymmh[16]; // High 16 bytes of each of 16 YMM registers (the low bytes
                    // are in FXSAVE.xmm for compatibility with SSE)
  uint64_t reserved3[16];
  MPXReg mpxr[4];   // MPX BNDREG state, containing 128-bit bound registers
  MPXCsr mpxc[2];   // MPX BNDCSR state, containing 64-bit BNDCFGU and
                    // BNDSTATUS registers
};
LLVM_PACKED_END

// Floating-point registers
union FPR {
  FXSAVE fxsave; // Generic floating-point registers.
  XSAVE xsave;   // x86 extended processor state.
};

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// Convenience function to combine YMM register data from XSAVE-style input.
inline YMMReg XStateToYMM(const void* xmm_bytes, const void* ymmh_bytes) {
  YMMReg ret;

  ::memcpy(ret.bytes, xmm_bytes, sizeof(XMMReg));
  ::memcpy(ret.bytes + sizeof(XMMReg), ymmh_bytes, sizeof(YMMHReg));

  return ret;
}

// Convenience function to copy YMM register data into XSAVE-style output.
inline void YMMToXState(const YMMReg& input, void* xmm_bytes, void* ymmh_bytes) {
  ::memcpy(xmm_bytes, input.bytes, sizeof(XMMReg));
  ::memcpy(ymmh_bytes, input.bytes + sizeof(XMMReg), sizeof(YMMHReg));
}

uint16_t AbridgedToFullTagWord(uint8_t abridged_tw, uint16_t sw,
                               llvm::ArrayRef<MMSReg> st_regs);
uint8_t FullToAbridgedTagWord(uint16_t tw);

namespace x86_register_info {
extern uint32_t g_contained_eax[];
extern uint32_t g_contained_ebx[];
extern uint32_t g_contained_ecx[];
extern uint32_t g_contained_edx[];
extern uint32_t g_contained_edi[];
extern uint32_t g_contained_esi[];
extern uint32_t g_contained_ebp[];
extern uint32_t g_contained_esp[];

extern uint32_t g_invalidate_eax[];
extern uint32_t g_invalidate_ebx[];
extern uint32_t g_invalidate_ecx[];
extern uint32_t g_invalidate_edx[];
extern uint32_t g_invalidate_edi[];
extern uint32_t g_invalidate_esi[];
extern uint32_t g_invalidate_ebp[];
extern uint32_t g_invalidate_esp[];

extern uint32_t g_contained_rax[];
extern uint32_t g_contained_rbx[];
extern uint32_t g_contained_rcx[];
extern uint32_t g_contained_rdx[];
extern uint32_t g_contained_rdi[];
extern uint32_t g_contained_rsi[];
extern uint32_t g_contained_rbp[];
extern uint32_t g_contained_rsp[];
extern uint32_t g_contained_r8[];
extern uint32_t g_contained_r9[];
extern uint32_t g_contained_r10[];
extern uint32_t g_contained_r11[];
extern uint32_t g_contained_r12[];
extern uint32_t g_contained_r13[];
extern uint32_t g_contained_r14[];
extern uint32_t g_contained_r15[];

extern uint32_t g_invalidate_rax[];
extern uint32_t g_invalidate_rbx[];
extern uint32_t g_invalidate_rcx[];
extern uint32_t g_invalidate_rdx[];
extern uint32_t g_invalidate_rdi[];
extern uint32_t g_invalidate_rsi[];
extern uint32_t g_invalidate_rbp[];
extern uint32_t g_invalidate_rsp[];
extern uint32_t g_invalidate_r8[];
extern uint32_t g_invalidate_r9[];
extern uint32_t g_invalidate_r10[];
extern uint32_t g_invalidate_r11[];
extern uint32_t g_invalidate_r12[];
extern uint32_t g_invalidate_r13[];
extern uint32_t g_invalidate_r14[];
extern uint32_t g_invalidate_r15[];

extern uint32_t g_contained_fip[];
extern uint32_t g_contained_fdp[];

extern uint32_t g_invalidate_fip[];
extern uint32_t g_invalidate_fdp[];

extern uint32_t g_contained_st0_32[];
extern uint32_t g_contained_st1_32[];
extern uint32_t g_contained_st2_32[];
extern uint32_t g_contained_st3_32[];
extern uint32_t g_contained_st4_32[];
extern uint32_t g_contained_st5_32[];
extern uint32_t g_contained_st6_32[];
extern uint32_t g_contained_st7_32[];

extern uint32_t g_invalidate_st0_32[];
extern uint32_t g_invalidate_st1_32[];
extern uint32_t g_invalidate_st2_32[];
extern uint32_t g_invalidate_st3_32[];
extern uint32_t g_invalidate_st4_32[];
extern uint32_t g_invalidate_st5_32[];
extern uint32_t g_invalidate_st6_32[];
extern uint32_t g_invalidate_st7_32[];

extern uint32_t g_contained_st0_64[];
extern uint32_t g_contained_st1_64[];
extern uint32_t g_contained_st2_64[];
extern uint32_t g_contained_st3_64[];
extern uint32_t g_contained_st4_64[];
extern uint32_t g_contained_st5_64[];
extern uint32_t g_contained_st6_64[];
extern uint32_t g_contained_st7_64[];

extern uint32_t g_invalidate_st0_64[];
extern uint32_t g_invalidate_st1_64[];
extern uint32_t g_invalidate_st2_64[];
extern uint32_t g_invalidate_st3_64[];
extern uint32_t g_invalidate_st4_64[];
extern uint32_t g_invalidate_st5_64[];
extern uint32_t g_invalidate_st6_64[];
extern uint32_t g_invalidate_st7_64[];
} // namespace x86_register_info

} // namespace lldb_private

#endif
