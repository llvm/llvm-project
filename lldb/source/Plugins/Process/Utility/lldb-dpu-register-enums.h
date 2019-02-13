//===-- lldb-dpu-register-enums.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_dpu_register_enums_h
#define lldb_dpu_register_enums_h

namespace lldb_private {
// LLDB register codes (e.g. RegisterKind == eRegisterKindLLDB)

//---------------------------------------------------------------------------
// Internal codes for all DPU registers.
//---------------------------------------------------------------------------
enum {
  r0_dpu,
  r1_dpu,
  r2_dpu,
  r3_dpu,
  r4_dpu,
  r5_dpu,
  r6_dpu,
  r7_dpu,
  r8_dpu,
  r9_dpu,
  r10_dpu,
  r11_dpu,
  r12_dpu,
  r13_dpu,
  r14_dpu,
  r15_dpu,
  r16_dpu,
  r17_dpu,
  r18_dpu,
  r19_dpu,
  r20_dpu,
  r21_dpu,
  r22_dpu,
  r23_dpu,
  sp_dpu = r22_dpu,
  lr_dpu = r23_dpu,
  pc_dpu,

  k_num_registers_dpu,
};
} // namespace lldb_private

#endif // #ifndef lldb_dpu_register_enums_h
