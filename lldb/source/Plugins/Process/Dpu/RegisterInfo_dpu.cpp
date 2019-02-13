//===-- RegisterInfo_dpu.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include <cassert>
#include <stddef.h>
#include <vector>

#include "lldb/lldb-defines.h"
#include "llvm/Support/Compiler.h"

//#include "lldb/lldb-enumerations.h"
//#include "lldb/lldb-private.h"

#include "RegisterInfo_dpu.h"

using namespace lldb;
using namespace lldb_private;

enum {
  dwarf_r0 = 0,
  dwarf_r1,
  dwarf_r2,
  dwarf_r3,
  dwarf_r4,
  dwarf_r5,
  dwarf_r6,
  dwarf_r7,
  dwarf_r8,
  dwarf_r9,
  dwarf_r10,
  dwarf_r11,
  dwarf_r12,
  dwarf_r13,
  dwarf_r14,
  dwarf_r15,
  dwarf_r16,
  dwarf_r17,
  dwarf_r18,
  dwarf_r19,
  dwarf_r20,
  dwarf_r21,
  dwarf_r22,
  dwarf_r23,
  dwarf_pc,
};
// The register numbers used in the eh_frame unwind information.
// Should be the same as DWARF register numbers.
enum {
  ehframe_r0 = dwarf_r0,
  ehframe_r1 = dwarf_r1,
  ehframe_r2 = dwarf_r2,
  ehframe_r3 = dwarf_r3,
  ehframe_r4 = dwarf_r4,
  ehframe_r5 = dwarf_r5,
  ehframe_r6 = dwarf_r6,
  ehframe_r7 = dwarf_r7,
  ehframe_r8 = dwarf_r8,
  ehframe_r9 = dwarf_r9,
  ehframe_r10 = dwarf_r10,
  ehframe_r11 = dwarf_r11,
  ehframe_r12 = dwarf_r12,
  ehframe_r13 = dwarf_r13,
  ehframe_r14 = dwarf_r14,
  ehframe_r15 = dwarf_r15,
  ehframe_r16 = dwarf_r16,
  ehframe_r17 = dwarf_r17,
  ehframe_r18 = dwarf_r18,
  ehframe_r19 = dwarf_r19,
  ehframe_r20 = dwarf_r20,
  ehframe_r21 = dwarf_r21,
  ehframe_r22 = dwarf_r22,
  ehframe_r23 = dwarf_r23,
  ehframe_pc = dwarf_pc,
};

enum {
  gpr_r0 = 0,
  gpr_r1,
  gpr_r2,
  gpr_r3,
  gpr_r4,
  gpr_r5,
  gpr_r6,
  gpr_r7,
  gpr_r8,
  gpr_r9,
  gpr_r10,
  gpr_r11,
  gpr_r12,
  gpr_r13,
  gpr_r14,
  gpr_r15,
  gpr_r16,
  gpr_r17,
  gpr_r18,
  gpr_r19,
  gpr_r20,
  gpr_r21,
  gpr_r22,
  gpr_r23,
  gpr_sp = gpr_r22,
  gpr_lr = gpr_r23,
  gpr_pc,
};

#define XSTR(s) STR(s)
#define STR(s) #s
#define DEFINE_GPR(idx, generic)                                               \
  {                                                                            \
    "r" STR(idx), nullptr, 4, ((idx)*4), eEncodingUint, eFormatHex,            \
        {ehframe_r##idx, dwarf_r##idx, LLDB_REGNUM_GENERIC_ARG1,               \
         LLDB_INVALID_REGNUM, gpr_r##idx},                                     \
        nullptr, nullptr, nullptr, 0                                           \
  }

static RegisterInfo g_register_infos_dpu[] = {
    //  NAME         ALT     SZ   OFFSET          ENCODING          FORMAT
    //  EH_FRAME             DWARF                GENERIC
    //  PROCESS PLUGIN       LLDB NATIVE      VALUE REGS      INVALIDATE REGS
    DEFINE_GPR(0, LLDB_REGNUM_GENERIC_ARG1),
    DEFINE_GPR(1, LLDB_REGNUM_GENERIC_ARG2),
    DEFINE_GPR(2, LLDB_REGNUM_GENERIC_ARG3),
    DEFINE_GPR(3, LLDB_REGNUM_GENERIC_ARG4),
    DEFINE_GPR(4, LLDB_REGNUM_GENERIC_ARG5),
    DEFINE_GPR(5, LLDB_REGNUM_GENERIC_ARG6),
    DEFINE_GPR(6, LLDB_REGNUM_GENERIC_ARG7),
    DEFINE_GPR(7, LLDB_REGNUM_GENERIC_ARG8),
    DEFINE_GPR(8, LLDB_REGNUM_GENERIC_ARG9),
    DEFINE_GPR(9, LLDB_REGNUM_GENERIC_ARG10),
    DEFINE_GPR(10, LLDB_REGNUM_GENERIC_ARG11),
    DEFINE_GPR(11, LLDB_REGNUM_GENERIC_ARG12),
    DEFINE_GPR(12, LLDB_REGNUM_GENERIC_ARG13),
    DEFINE_GPR(13, LLDB_REGNUM_GENERIC_ARG14),
    DEFINE_GPR(14, LLDB_REGNUM_GENERIC_ARG15),
    DEFINE_GPR(15, LLDB_REGNUM_GENERIC_ARG16),
    DEFINE_GPR(16, LLDB_INVALID_REGNUM),
    DEFINE_GPR(17, LLDB_INVALID_REGNUM),
    DEFINE_GPR(18, LLDB_INVALID_REGNUM),
    DEFINE_GPR(19, LLDB_INVALID_REGNUM),
    DEFINE_GPR(20, LLDB_INVALID_REGNUM),
    DEFINE_GPR(21, LLDB_INVALID_REGNUM),
    DEFINE_GPR(22, LLDB_REGNUM_GENERIC_SP),
    DEFINE_GPR(23, LLDB_REGNUM_GENERIC_RA),
    {"pc",
     nullptr,
     4,
     24 * 4,
     eEncodingUint,
     eFormatHex,
     {ehframe_pc, dwarf_pc, LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_REGNUM,
      gpr_pc},
     nullptr,
     nullptr,
     nullptr,
     0}};

const uint32_t k_register_infos_count =
    sizeof(g_register_infos_dpu) / sizeof(g_register_infos_dpu[0]);

const ArchSpec k_dpu_arch("dpu-upmem-dpurte");

RegisterInfo_dpu::RegisterInfo_dpu()
    : lldb_private::RegisterInfoInterface(k_dpu_arch) {}

size_t RegisterInfo_dpu::GetGPRSize() const {
  return sizeof(struct RegisterInfo_dpu::GPR);
}

const lldb_private::RegisterInfo *RegisterInfo_dpu::GetRegisterInfo() const {
  return g_register_infos_dpu;
}

uint32_t RegisterInfo_dpu::GetRegisterCount() const {
  return k_register_infos_count;
}
