//===-- MipsBaseInfo.h - Top level definitions for MIPS MC ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the Mips target useful for the compiler back-end and the MC libraries.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSBASEINFO_H
#define LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSBASEINFO_H

#include "MipsFixupKinds.h"
#include "MipsMCTargetDesc.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

/// MipsII - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace MipsII {
/// Target Operand Flag enum.
enum TOF {
  //===------------------------------------------------------------------===//
  // Mips Specific MachineOperand flags.

  MO_NO_FLAG,

  // Represents the offset into the global offset table at which
  // the address the relocation entry symbol resides during execution.
  MO_GOT,

  // Represents the offset into the global offset table at
  // which the address of a call site relocation entry symbol resides
  // during execution. This is different from the above since this flag
  // can only be present in call instructions.
  MO_GOT_CALL,

  // Represents the offset from the current gp value to be used
  // for the relocatable object file being produced.
  MO_GPREL,

  // Represents the hi or low part of an absolute symbol
  // address.
  MO_ABS_HI,
  MO_ABS_LO,

  // Represents the offset into the global offset table at which
  // the module ID and TSL block offset reside during execution (General
  // Dynamic TLS).
  MO_TLSGD,

  // Represents the offset into the global offset table at which
  // the module ID and TSL block offset reside during execution (Local
  // Dynamic TLS).
  MO_TLSLDM,
  MO_DTPREL_HI,
  MO_DTPREL_LO,

  // Represents the offset from the thread pointer (Initial
  // Exec TLS).
  MO_GOTTPREL,

  // Represents the hi and low part of the offset from
  // the thread pointer (Local Exec TLS).
  MO_TPREL_HI,
  MO_TPREL_LO,

  // N32/64 Flags.
  MO_GPOFF_HI,
  MO_GPOFF_LO,
  MO_GOT_DISP,
  MO_GOT_PAGE,
  MO_GOT_OFST,

  // Represents the highest or higher half word of a
  // 64-bit symbol address.
  MO_HIGHER,
  MO_HIGHEST,

  // Relocations used for large GOTs.
  MO_GOT_HI16,
  MO_GOT_LO16,
  MO_CALL_HI16,
  MO_CALL_LO16,

  // Helper operand used to generate R_MIPS_JALR
  MO_JALR,

  // On a symbol operand "FOO", this indicates that the
  // reference is actually to the "__imp_FOO" symbol.  This is used for
  // dllimport linkage on windows.
  MO_DLLIMPORT = 0x20,
};

enum {
  //===------------------------------------------------------------------===//
  // Instruction encodings.  These are the standard/most common forms for
  // Mips instructions.
  //

  // This represents an instruction that is a pseudo instruction
  // or one that has not been implemented yet.  It is illegal to code generate
  // it, but tolerated for intermediate implementation stages.
  Pseudo = 0,

  // This form is for instructions of the format R.
  FrmR = 1,
  // This form is for instructions of the format I.
  FrmI = 2,
  // This form is for instructions of the format J.
  FrmJ = 3,
  // This form is for instructions of the format FR.
  FrmFR = 4,
  // This form is for instructions of the format FI.
  FrmFI = 5,
  // This form is for instructions that have no specific format.
  FrmOther = 6,

  FormMask = 15,
  // Instruction is a Control Transfer Instruction.
  IsCTI = 1 << 4,
  // Instruction has a forbidden slot.
  HasForbiddenSlot = 1 << 5,
  //  Instruction uses an $fcc<x> register.
  HasFCCRegOperand = 1 << 6

};

enum OperandType : unsigned {
  OPERAND_FIRST_MIPS_MEM_IMM = MCOI::OPERAND_FIRST_TARGET,
  OPERAND_MEM_SIMM9 = OPERAND_FIRST_MIPS_MEM_IMM,
  OPERAND_LAST_MIPS_MEM_IMM = OPERAND_MEM_SIMM9
};

static inline unsigned getFormat(uint64_t TSFlags) {
  return TSFlags & FormMask;
}
} // namespace MipsII

inline static MCRegister getMSARegFromFReg(MCRegister Reg) {
  if (Reg >= Mips::F0 && Reg <= Mips::F31)
    return Reg - Mips::F0 + Mips::W0;
  else if (Reg >= Mips::D0_64 && Reg <= Mips::D31_64)
    return Reg - Mips::D0_64 + Mips::W0;
  else
    return MCRegister();
}
} // namespace llvm

#endif
