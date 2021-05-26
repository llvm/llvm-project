//===-- RegisterInfos_mips64.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>

#include "lldb/Core/dwarf.h"
#include "llvm/Support/Compiler.h"


#ifdef DECLARE_REGISTER_INFOS_MIPS64_STRUCT

// Computes the offset of the given GPR in the user data area.
#define GPR_OFFSET(regname) (LLVM_EXTENSION offsetof(GPR_freebsd_mips, regname))

// RegisterKind: EHFrame, DWARF, Generic, Process Plugin, LLDB

// Note that the size and offset will be updated by platform-specific classes.
#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)                       \
  {                                                                            \
    #reg, alt, sizeof(((GPR_freebsd_mips *) 0)->reg),                          \
                      GPR_OFFSET(reg), eEncodingUint, eFormatHex,              \
                                 {kind1, kind2, kind3, kind4,                  \
                                  gpr_##reg##_mips64 },                        \
                                  NULL, NULL, NULL, 0                          \
  }

static RegisterInfo g_register_infos_mips64[] = {
// General purpose registers.            EH_Frame,                  DWARF,
// Generic,    Process Plugin
    DEFINE_GPR(zero, "r0", dwarf_zero_mips64, dwarf_zero_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r1, nullptr, dwarf_r1_mips64, dwarf_r1_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r2, nullptr, dwarf_r2_mips64, dwarf_r2_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r3, nullptr, dwarf_r3_mips64, dwarf_r3_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r4, nullptr, dwarf_r4_mips64, dwarf_r4_mips64,
               LLDB_REGNUM_GENERIC_ARG1, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r5, nullptr, dwarf_r5_mips64, dwarf_r5_mips64,
               LLDB_REGNUM_GENERIC_ARG2, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r6, nullptr, dwarf_r6_mips64, dwarf_r6_mips64,
               LLDB_REGNUM_GENERIC_ARG3, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r7, nullptr, dwarf_r7_mips64, dwarf_r7_mips64,
               LLDB_REGNUM_GENERIC_ARG4, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r8, nullptr, dwarf_r8_mips64, dwarf_r8_mips64,
               LLDB_REGNUM_GENERIC_ARG5, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r9, nullptr, dwarf_r9_mips64, dwarf_r9_mips64,
               LLDB_REGNUM_GENERIC_ARG6, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r10, nullptr, dwarf_r10_mips64, dwarf_r10_mips64,
               LLDB_REGNUM_GENERIC_ARG7, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r11, nullptr, dwarf_r11_mips64, dwarf_r11_mips64,
               LLDB_REGNUM_GENERIC_ARG8, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r12, nullptr, dwarf_r12_mips64, dwarf_r12_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r13, nullptr, dwarf_r13_mips64, dwarf_r13_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r14, nullptr, dwarf_r14_mips64, dwarf_r14_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r15, nullptr, dwarf_r15_mips64, dwarf_r15_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r16, nullptr, dwarf_r16_mips64, dwarf_r16_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r17, nullptr, dwarf_r17_mips64, dwarf_r17_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r18, nullptr, dwarf_r18_mips64, dwarf_r18_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r19, nullptr, dwarf_r19_mips64, dwarf_r19_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r20, nullptr, dwarf_r20_mips64, dwarf_r20_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r21, nullptr, dwarf_r21_mips64, dwarf_r21_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r22, nullptr, dwarf_r22_mips64, dwarf_r22_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r23, nullptr, dwarf_r23_mips64, dwarf_r23_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r24, nullptr, dwarf_r24_mips64, dwarf_r24_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r25, nullptr, dwarf_r25_mips64, dwarf_r25_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r26, nullptr, dwarf_r26_mips64, dwarf_r26_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r27, nullptr, dwarf_r27_mips64, dwarf_r27_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(gp, "r28", dwarf_gp_mips64, dwarf_gp_mips64, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(sp, "r29", dwarf_sp_mips64, dwarf_sp_mips64,
               LLDB_REGNUM_GENERIC_SP, LLDB_INVALID_REGNUM),
    DEFINE_GPR(r30, nullptr, dwarf_r30_mips64, dwarf_r30_mips64,
               LLDB_REGNUM_GENERIC_FP, LLDB_INVALID_REGNUM),
    DEFINE_GPR(ra, "r31", dwarf_ra_mips64, dwarf_ra_mips64,
               LLDB_REGNUM_GENERIC_RA, LLDB_INVALID_REGNUM),
    DEFINE_GPR(sr, nullptr, dwarf_sr_mips64, dwarf_sr_mips64,
               LLDB_REGNUM_GENERIC_FLAGS, LLDB_INVALID_REGNUM),
    DEFINE_GPR(mullo, nullptr, dwarf_lo_mips64, dwarf_lo_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(mulhi, nullptr, dwarf_hi_mips64, dwarf_hi_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(badvaddr, nullptr, dwarf_bad_mips64, dwarf_bad_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(cause, nullptr, dwarf_cause_mips64, dwarf_cause_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(pc, "pc", dwarf_pc_mips64, dwarf_pc_mips64,
               LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_REGNUM),
    DEFINE_GPR(ic, nullptr, dwarf_ic_mips64, dwarf_ic_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
    DEFINE_GPR(dummy, nullptr, dwarf_dummy_mips64, dwarf_dummy_mips64,
               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),
};

static_assert((sizeof(g_register_infos_mips64) /
               sizeof(g_register_infos_mips64[0])) == k_num_registers_mips64,
              "g_register_infos_mips64 has wrong number of register infos");

#undef DEFINE_GPR
#undef DEFINE_GPR_INFO
#undef DEFINE_FPR
#undef DEFINE_FPR_INFO
#undef DEFINE_MSA
#undef DEFINE_MSA_INFO
#undef GPR_OFFSET
#undef FPR_OFFSET
#undef MSA_OFFSET

#endif // DECLARE_REGISTER_INFOS_MIPS64_STRUCT
