//===-- DPUCondCodes.h - DPU Conditions -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Mapping of the standard ISD conditional codes with known DPU conditional
// codes.
//
// The DPU provides two types of conditional codes:
//  - Binary conditionals: to compare two items together
//  - Unary conditionals: where the comparison is implicitly with a null operand
//
// Notice that the DPU is not able to map everything in this.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_DPUCONDCODES_H
#define LLVM_LIB_TARGET_DPU_DPUCONDCODES_H

#include "llvm/CodeGen/ISDOpcodes.h"

namespace llvm {
namespace DPU {
enum BinaryCondCode {
  COND_EQ = 0,
  COND_NEQ,
  COND_GTS,
  COND_GES,
  COND_LTS,
  COND_LES,
  COND_GTU,
  COND_LTU,
  COND_GEU,
  COND_LEU,
  // Reserved to report errors
  COND_UNDEF_BINARY
};

enum UnaryCondCode {
  COND_Z = 0,
  COND_NZ,
  COND_PL,
  COND_MI,
  COND_GT0S,
  COND_GE0S,
  COND_LT0S,
  COND_LE0S,
  COND_GT0U,
  COND_LT0U,
  COND_GE0U,
  COND_LE0U,
  COND_TRUE,
  COND_SZ,
  COND_SNZ,
  COND_SPL,
  COND_SMI,
  // Reserved to report errors
  COND_UNDEF_UNARY
};

enum AddedISDCondCode {
  ISD_COND_SZ = 0x10000,
  ISD_COND_SNZ = 0x10001,
  ISD_COND_SPL = 0x10010,
  ISD_COND_SMI = 0x10011,
};
} // namespace DPU

class DpuBinaryCondCode {
public:
  // @param FromCC a standard conditional code
  // @return The corresponding DPU binary conditional, COND_UNDEF_BINARY if the
  // input is not handled.
  DPU::BinaryCondCode FromIsdCondCode(ISD::CondCode FromCC) {
    switch (FromCC) {
    default:
      return DPU::COND_UNDEF_BINARY;

    case ISD::CondCode::SETEQ:
    case ISD::CondCode::SETOEQ:
    case ISD::CondCode::SETUEQ:
      return DPU::COND_EQ;

    case ISD::CondCode::SETNE:
    case ISD::CondCode::SETONE:
    case ISD::CondCode::SETUNE:
      return DPU::COND_NEQ;

    case ISD::CondCode::SETOGT:
    case ISD::CondCode::SETGT:
      return DPU::COND_GTS;

    case ISD::CondCode::SETOGE:
    case ISD::CondCode::SETGE:
      return DPU::COND_GES;

    case ISD::CondCode::SETOLT:
    case ISD::CondCode::SETLT:
      return DPU::COND_LTS;

    case ISD::CondCode::SETOLE:
    case ISD::CondCode::SETLE:
      return DPU::COND_LES;

    case ISD::CondCode::SETUGT:
      return DPU::COND_GTU;

    case ISD::CondCode::SETULT:
      return DPU::COND_LTU;

    case ISD::CondCode::SETUGE:
      return DPU::COND_GEU;

    case ISD::CondCode::SETULE:
      return DPU::COND_LEU;
    }
  }

  // @param FromCC a standard conditional code
  // @return Whether this conditional code is known by the DPU.
  bool CanMapIsdCondCode(ISD::CondCode FromCC) {
    return FromIsdCondCode(FromCC) != DPU::COND_UNDEF_BINARY;
  }

  // @param BCC A binary conditional code, as returned by FromIsdCondCode
  // @return The keyword part of an asm instruction representing this condition.
  const char *AsKeyword(DPU::BinaryCondCode BCC) {
    switch (BCC) {
    case DPU::COND_EQ:
      return "eq";

    case DPU::COND_NEQ:
      return "neq";

    case DPU::COND_GTS:
      return "gts";

    case DPU::COND_GES:
      return "ges";

    case DPU::COND_LTS:
      return "lts";

    case DPU::COND_LES:
      return "les";

    case DPU::COND_GTU:
      return "gtu";

    case DPU::COND_LTU:
      return "ltu";

    case DPU::COND_GEU:
      return "geu";

    case DPU::COND_LEU:
      return "leu";

    default:
      return "xxx";
    }
  }
};

class DpuUnaryCondCode {
public:
  // @param FromCC a standard conditional code
  // @return The corresponding DPU binary conditional, COND_UNDEF_BINARY if the
  // input is not handled.
  DPU::UnaryCondCode FromIsdCondCode(ISD::CondCode FromCC) {
    switch (FromCC) {
    case ISD::CondCode::SETEQ:
    case ISD::CondCode::SETOEQ:
    case ISD::CondCode::SETUEQ:
      return DPU::COND_Z;

    case ISD::CondCode::SETNE:
    case ISD::CondCode::SETONE:
    case ISD::CondCode::SETUNE:
      return DPU::COND_NZ;

    case ISD::CondCode::SETOGT:
    case ISD::CondCode::SETGT:
      return DPU::COND_GT0S;

    case ISD::CondCode::SETUGT:
      return DPU::COND_GT0U;

    case ISD::CondCode::SETUGE:
      return DPU::COND_GE0U;

    case ISD::CondCode::SETOGE:
    case ISD::CondCode::SETGE:
      return DPU::COND_PL;

    case ISD::CondCode::SETOLT:
    case ISD::CondCode::SETLT:
      return DPU::COND_MI;

    case ISD::CondCode::SETULT:
      return DPU::COND_LT0U;

    case ISD::CondCode::SETULE:
      return DPU::COND_LE0U;

    case ISD::CondCode::SETOLE:
    case ISD::CondCode::SETLE:
      return DPU::COND_LE0S;

    case ISD::CondCode::SETTRUE:
    case ISD::CondCode::SETTRUE2:
      return DPU::COND_TRUE;

    default:
      switch ((DPU::AddedISDCondCode)FromCC) {
      case DPU::AddedISDCondCode::ISD_COND_SZ:
        return DPU::COND_SZ;
      case DPU::AddedISDCondCode::ISD_COND_SNZ:
        return DPU::COND_SNZ;
      case DPU::AddedISDCondCode::ISD_COND_SPL:
        return DPU::COND_SPL;
      case DPU::AddedISDCondCode::ISD_COND_SMI:
        return DPU::COND_SMI;
      default:
        break;
      }
      return DPU::COND_UNDEF_UNARY;
    }
  }

  // @param FromCC a standard conditional code
  // @return Whether this conditional code is known by the DPU.
  bool CanMapIsdCondCode(ISD::CondCode FromCC, unsigned int ForOpcode) {
    DPU::UnaryCondCode DPUCondCode = FromIsdCondCode(FromCC);
    if (DPUCondCode == DPU::COND_UNDEF_UNARY)
      return false;
    switch (ForOpcode) {
    case ISD::OR:
    case ISD::AND:
    case ISD::XOR:
      switch (DPUCondCode) {
      case DPU::COND_Z:
      case DPU::COND_NZ:
      case DPU::COND_PL:
      case DPU::COND_MI:
        return true;

      default:
        return false;
      }

    case ISD::ADD:
      switch (DPUCondCode) {
      case DPU::COND_Z:
      case DPU::COND_NZ:
      case DPU::COND_PL:
      case DPU::COND_MI:
        return true;

      case DPU::COND_GT0S:
      case DPU::COND_GE0S:
      case DPU::COND_LT0S:
      case DPU::COND_LE0S:
      case DPU::COND_GT0U:
      case DPU::COND_LT0U:
      case DPU::COND_GE0U:
      case DPU::COND_LE0U:
        /* todo: we can use pl/mi in some cases, but we need to be
         * able to provide the condition to the rest of the compiler... */

      default:
        return false;
      }

    case ISD::SUB:
      switch (DPUCondCode) {
      case DPU::COND_Z:
      case DPU::COND_NZ:
      case DPU::COND_PL:
      case DPU::COND_MI:
      case DPU::COND_GT0S:
      case DPU::COND_GE0S:
      case DPU::COND_LT0S:
      case DPU::COND_LE0S:
      case DPU::COND_GT0U:
      case DPU::COND_LT0U:
      case DPU::COND_GE0U:
      case DPU::COND_LE0U:
        return true;

      default:
        return false;
      }
    }
  }

  // @param BCC A binary conditional code, as returned by FromIsdCondCode
  // @return The keyword part of an asm instruction representing this condition.
  const char *AsKeyword(DPU::UnaryCondCode BCC) {
    switch (BCC) {
    case DPU::COND_Z:
      return "z";

    case DPU::COND_NZ:
      return "nz";

    case DPU::COND_PL:
      return "pl";

    case DPU::COND_MI:
      return "mi";

    case DPU::COND_GT0S:
      return "gts";

    case DPU::COND_GE0S:
      return "ges";

    case DPU::COND_LT0S:
      return "lts";

    case DPU::COND_LE0S:
      return "les";

    case DPU::COND_GT0U:
      return "gtu";

    case DPU::COND_LT0U:
      return "ltu";

    case DPU::COND_GE0U:
      return "geu";

    case DPU::COND_LE0U:
      return "leu";

    case DPU::COND_TRUE:
      return "true";

    case DPU::COND_SZ:
      return "sz";

    case DPU::COND_SNZ:
      return "snz";

    case DPU::COND_SPL:
      return "spl";

    case DPU::COND_SMI:
      return "smi";

    default:
      return "xxx";
    }
  }
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_DPU_DPUCONDCODES_H
