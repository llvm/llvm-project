//===-- VE.h - Top-level interface for VE representation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// VE back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_VE_VE_H
#define LLVM_LIB_TARGET_VE_VE_H

#include "MCTargetDesc/VEMCTargetDesc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class FunctionPass;
class VETargetMachine;
class formatted_raw_ostream;
class AsmPrinter;
class MCInst;
class MachineInstr;

FunctionPass *createVEISelDag(VETargetMachine &TM);
FunctionPass *createVEPromoteToI1Pass();

void LowerVEMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                 AsmPrinter &AP);
} // namespace llvm

namespace llvm {
// Enums corresponding to VE condition codes, both icc's and fcc's.  These
// values must be kept in sync with the ones in the .td file.
namespace VECC {
enum CondCode {
  // Integer comparison
  CC_IG =  0,  // Greater
  CC_IL =  1,  // Less
  CC_INE = 2, // Not Equal
  CC_IEQ = 3, // Equal
  CC_IGE = 4, // Greater or Equal
  CC_ILE = 5, // Less or Equal

  // Floating point comparison
  CC_AF =     0 + 6, // Never
  CC_G =      1 + 6, // Greater
  CC_L =      2 + 6, // Less
  CC_NE =     3 + 6, // Not Equal
  CC_EQ =     4 + 6, // Equal
  CC_GE =     5 + 6, // Greater or Equal
  CC_LE =     6 + 6, // Less or Equal
  CC_NUM =    7 + 6, // Number
  CC_NAN =    8 + 6, // NaN
  CC_GNAN =   9 + 6, // Greater or NaN
  CC_LNAN =  10 + 6, // Less or NaN
  CC_NENAN = 11 + 6, // Not Equal or NaN
  CC_EQNAN = 12 + 6, // Equal or NaN
  CC_GENAN = 13 + 6, // Greater or Equal or NaN
  CC_LENAN = 14 + 6, // Less or Equal or NaN
  CC_AT =    15 + 6, // Always
};
}

inline static const char *VECondCodeToString(VECC::CondCode CC) {
  switch (CC) {
  case VECC::CC_IG:    return "gt";
  case VECC::CC_IL:    return "lt";
  case VECC::CC_INE:   return "ne";
  case VECC::CC_IEQ:   return "eq";
  case VECC::CC_IGE:   return "ge";
  case VECC::CC_ILE:   return "le";
  case VECC::CC_AF:    return "af";
  case VECC::CC_G:     return "gt";
  case VECC::CC_L:     return "lt";
  case VECC::CC_NE:    return "ne";
  case VECC::CC_EQ:    return "eq";
  case VECC::CC_GE:    return "ge";
  case VECC::CC_LE:    return "le";
  case VECC::CC_NUM:   return "num";
  case VECC::CC_NAN:   return "nan";
  case VECC::CC_GNAN:  return "gtnan";
  case VECC::CC_LNAN:  return "ltnan";
  case VECC::CC_NENAN: return "nenan";
  case VECC::CC_EQNAN: return "eqnan";
  case VECC::CC_GENAN: return "genan";
  case VECC::CC_LENAN: return "lenan";
  case VECC::CC_AT:    return "at";
  }
  llvm_unreachable("Invalid cond code");
}

inline unsigned M0(unsigned Val) { return Val + 64; }
inline unsigned M1(unsigned Val) { return Val; }

} // namespace llvm
#endif
