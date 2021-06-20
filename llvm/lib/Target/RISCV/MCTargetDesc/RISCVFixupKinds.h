//===-- RISCVFixupKinds.h - RISCV Specific Fixup Entries --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVFIXUPKINDS_H
#define LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

#undef RISCV

namespace llvm {
namespace RISCV {
enum Fixups {
  // 20-bit fixup corresponding to %hi(foo) for instructions like lui
  fixup_riscv_hi20 = FirstTargetFixupKind,
  // 12-bit fixup corresponding to %lo(foo) for instructions like addi
  fixup_riscv_lo12_i,
  // 12-bit fixup corresponding to %lo(foo) for the S-type store instructions
  fixup_riscv_lo12_s,
  // 20-bit fixup corresponding to %pcrel_hi(foo) for instructions like auipc
  fixup_riscv_pcrel_hi20,
  // 12-bit fixup corresponding to %pcrel_lo(foo) for instructions like addi
  fixup_riscv_pcrel_lo12_i,
  // 12-bit fixup corresponding to %pcrel_lo(foo) for the S-type store
  // instructions
  fixup_riscv_pcrel_lo12_s,
  // 20-bit fixup corresponding to %got_pcrel_hi(foo) for instructions like
  // auipc
  fixup_riscv_got_hi20,
  // 20-bit fixup corresponding to %tprel_hi(foo) for instructions like lui
  fixup_riscv_tprel_hi20,
  // 12-bit fixup corresponding to %tprel_lo(foo) for instructions like addi
  fixup_riscv_tprel_lo12_i,
  // 12-bit fixup corresponding to %tprel_lo(foo) for the S-type store
  // instructions
  fixup_riscv_tprel_lo12_s,
  // Fixup corresponding to %tprel_add(foo) for PseudoAddTPRel, used as a linker
  // hint
  fixup_riscv_tprel_add,
  // 20-bit fixup corresponding to %tls_ie_pcrel_hi(foo) for instructions like
  // auipc
  fixup_riscv_tls_got_hi20,
  // 20-bit fixup corresponding to %tls_gd_pcrel_hi(foo) for instructions like
  // auipc
  fixup_riscv_tls_gd_hi20,
  // 20-bit fixup for symbol references in the jal instruction
  fixup_riscv_jal,
  // 12-bit fixup for symbol references in the branch instructions
  fixup_riscv_branch,
  // 11-bit fixup for symbol references in the compressed jump instruction
  fixup_riscv_rvc_jump,
  // 8-bit fixup for symbol references in the compressed branch instruction
  fixup_riscv_rvc_branch,
  // Fixup representing a legacy no-pic function call attached to the auipc
  // instruction in a pair composed of adjacent auipc+jalr instructions.
  fixup_riscv_call,
  // Fixup representing a function call attached to the auipc instruction in a
  // pair composed of adjacent auipc+jalr instructions.
  fixup_riscv_call_plt,
  // Used to generate an R_RISCV_RELAX relocation, which indicates the linker
  // may relax the instruction pair.
  fixup_riscv_relax,
  // Used to generate an R_RISCV_ALIGN relocation, which indicates the linker
  // should fixup the alignment after linker relaxation.
  fixup_riscv_align,

  // Used as a sentinel, must be the last
  fixup_riscv_invalid,
  NumTargetFixupKinds = fixup_riscv_invalid - FirstTargetFixupKind
};
} // end namespace RISCV
} // end namespace llvm

#endif
