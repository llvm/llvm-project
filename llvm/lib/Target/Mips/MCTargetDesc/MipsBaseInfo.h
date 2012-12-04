//===-- MipsBaseInfo.h - Top level definitions for MIPS MC ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the Mips target useful for the compiler back-end and the MC libraries.
//
//===----------------------------------------------------------------------===//
#ifndef MIPSBASEINFO_H
#define MIPSBASEINFO_H

#include "MipsFixupKinds.h"
#include "MipsMCTargetDesc.h"
#include "llvm/MC/MCExpr.h"
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

    /// MO_GOT16 - Represents the offset into the global offset table at which
    /// the address the relocation entry symbol resides during execution.
    MO_GOT16,
    MO_GOT,

    /// MO_GOT_CALL - Represents the offset into the global offset table at
    /// which the address of a call site relocation entry symbol resides
    /// during execution. This is different from the above since this flag
    /// can only be present in call instructions.
    MO_GOT_CALL,

    /// MO_GPREL - Represents the offset from the current gp value to be used
    /// for the relocatable object file being produced.
    MO_GPREL,

    /// MO_ABS_HI/LO - Represents the hi or low part of an absolute symbol
    /// address.
    MO_ABS_HI,
    MO_ABS_LO,

    /// MO_TLSGD - Represents the offset into the global offset table at which
    // the module ID and TSL block offset reside during execution (General
    // Dynamic TLS).
    MO_TLSGD,

    /// MO_TLSLDM - Represents the offset into the global offset table at which
    // the module ID and TSL block offset reside during execution (Local
    // Dynamic TLS).
    MO_TLSLDM,
    MO_DTPREL_HI,
    MO_DTPREL_LO,

    /// MO_GOTTPREL - Represents the offset from the thread pointer (Initial
    // Exec TLS).
    MO_GOTTPREL,

    /// MO_TPREL_HI/LO - Represents the hi and low part of the offset from
    // the thread pointer (Local Exec TLS).
    MO_TPREL_HI,
    MO_TPREL_LO,

    // N32/64 Flags.
    MO_GPOFF_HI,
    MO_GPOFF_LO,
    MO_GOT_DISP,
    MO_GOT_PAGE,
    MO_GOT_OFST,

    /// MO_HIGHER/HIGHEST - Represents the highest or higher half word of a
    /// 64-bit symbol address.
    MO_HIGHER,
    MO_HIGHEST,

    /// MO_GOT_HI16/LO16, MO_CALL_HI16/LO16 - Relocations used for large GOTs.
    MO_GOT_HI16,
    MO_GOT_LO16,
    MO_CALL_HI16,
    MO_CALL_LO16
  };

  enum {
    //===------------------------------------------------------------------===//
    // Instruction encodings.  These are the standard/most common forms for
    // Mips instructions.
    //

    // Pseudo - This represents an instruction that is a pseudo instruction
    // or one that has not been implemented yet.  It is illegal to code generate
    // it, but tolerated for intermediate implementation stages.
    Pseudo   = 0,

    /// FrmR - This form is for instructions of the format R.
    FrmR  = 1,
    /// FrmI - This form is for instructions of the format I.
    FrmI  = 2,
    /// FrmJ - This form is for instructions of the format J.
    FrmJ  = 3,
    /// FrmFR - This form is for instructions of the format FR.
    FrmFR = 4,
    /// FrmFI - This form is for instructions of the format FI.
    FrmFI = 5,
    /// FrmOther - This form is for instructions that have no specific format.
    FrmOther = 6,

    FormMask = 15
  };
}


/// getMipsRegisterNumbering - Given the enum value for some register,
/// return the number that it corresponds to.
inline static unsigned getMipsRegisterNumbering(unsigned RegEnum)
{
  switch (RegEnum) {
  case Mips::ZERO: case Mips::ZERO_64: case Mips::F0: case Mips::D0_64:
  case Mips::D0:   case Mips::FCC0:    case Mips::AC0:
    return 0;
  case Mips::AT: case Mips::AT_64: case Mips::F1: case Mips::D1_64:
  case Mips::AC1:
    return 1;
  case Mips::V0: case Mips::V0_64: case Mips::F2: case Mips::D2_64:
  case Mips::D1: case Mips::AC2:
    return 2;
  case Mips::V1: case Mips::V1_64: case Mips::F3: case Mips::D3_64:
  case Mips::AC3:
    return 3;
  case Mips::A0: case Mips::A0_64: case Mips::F4: case Mips::D4_64:
  case Mips::D2:
    return 4;
  case Mips::A1: case Mips::A1_64: case Mips::F5: case Mips::D5_64:
    return 5;
  case Mips::A2: case Mips::A2_64: case Mips::F6: case Mips::D6_64:
  case Mips::D3:
    return 6;
  case Mips::A3: case Mips::A3_64: case Mips::F7: case Mips::D7_64:
    return 7;
  case Mips::T0: case Mips::T0_64: case Mips::F8: case Mips::D8_64:
  case Mips::D4:
    return 8;
  case Mips::T1: case Mips::T1_64: case Mips::F9: case Mips::D9_64:
    return 9;
  case Mips::T2: case Mips::T2_64: case Mips::F10: case Mips::D10_64:
  case Mips::D5:
    return 10;
  case Mips::T3: case Mips::T3_64: case Mips::F11: case Mips::D11_64:
    return 11;
  case Mips::T4: case Mips::T4_64: case Mips::F12: case Mips::D12_64:
  case Mips::D6:
    return 12;
  case Mips::T5: case Mips::T5_64: case Mips::F13: case Mips::D13_64:
    return 13;
  case Mips::T6: case Mips::T6_64: case Mips::F14: case Mips::D14_64:
  case Mips::D7:
    return 14;
  case Mips::T7: case Mips::T7_64: case Mips::F15: case Mips::D15_64:
    return 15;
  case Mips::S0: case Mips::S0_64: case Mips::F16: case Mips::D16_64:
  case Mips::D8:
    return 16;
  case Mips::S1: case Mips::S1_64: case Mips::F17: case Mips::D17_64:
    return 17;
  case Mips::S2: case Mips::S2_64: case Mips::F18: case Mips::D18_64:
  case Mips::D9:
    return 18;
  case Mips::S3: case Mips::S3_64: case Mips::F19: case Mips::D19_64:
    return 19;
  case Mips::S4: case Mips::S4_64: case Mips::F20: case Mips::D20_64:
  case Mips::D10:
    return 20;
  case Mips::S5: case Mips::S5_64: case Mips::F21: case Mips::D21_64:
    return 21;
  case Mips::S6: case Mips::S6_64: case Mips::F22: case Mips::D22_64:
  case Mips::D11:
    return 22;
  case Mips::S7: case Mips::S7_64: case Mips::F23: case Mips::D23_64:
    return 23;
  case Mips::T8: case Mips::T8_64: case Mips::F24: case Mips::D24_64:
  case Mips::D12:
    return 24;
  case Mips::T9: case Mips::T9_64: case Mips::F25: case Mips::D25_64:
    return 25;
  case Mips::K0: case Mips::K0_64: case Mips::F26: case Mips::D26_64:
  case Mips::D13:
    return 26;
  case Mips::K1: case Mips::K1_64: case Mips::F27: case Mips::D27_64:
    return 27;
  case Mips::GP: case Mips::GP_64: case Mips::F28: case Mips::D28_64:
  case Mips::D14:
    return 28;
  case Mips::SP: case Mips::SP_64: case Mips::F29: case Mips::D29_64:
  case Mips::HWR29:
    return 29;
  case Mips::FP: case Mips::FP_64: case Mips::F30: case Mips::D30_64:
  case Mips::D15:
    return 30;
  case Mips::RA: case Mips::RA_64: case Mips::F31: case Mips::D31_64:
    return 31;
  default: llvm_unreachable("Unknown register number!");
  }
}

inline static std::pair<const MCSymbolRefExpr*, int64_t>
MipsGetSymAndOffset(const MCFixup &Fixup) {
  MCFixupKind FixupKind = Fixup.getKind();

  if ((FixupKind < FirstTargetFixupKind) ||
      (FixupKind >= MCFixupKind(Mips::LastTargetFixupKind)))
    return std::make_pair((const MCSymbolRefExpr*)0, (int64_t)0);

  const MCExpr *Expr = Fixup.getValue();
  MCExpr::ExprKind Kind = Expr->getKind();

  if (Kind == MCExpr::Binary) {
    const MCBinaryExpr *BE = static_cast<const MCBinaryExpr*>(Expr);
    const MCExpr *LHS = BE->getLHS();
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(BE->getRHS());

    if ((LHS->getKind() != MCExpr::SymbolRef) || !CE)
      return std::make_pair((const MCSymbolRefExpr*)0, (int64_t)0);

    return std::make_pair(cast<MCSymbolRefExpr>(LHS), CE->getValue());
  }

  if (Kind != MCExpr::SymbolRef)
    return std::make_pair((const MCSymbolRefExpr*)0, (int64_t)0);

  return std::make_pair(cast<MCSymbolRefExpr>(Expr), 0);
}
}

#endif
