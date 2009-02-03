//===-- ARM.h - Top-level interface for ARM representation---- --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// ARM back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_ARM_H
#define TARGET_ARM_H

#include <cassert>

namespace llvm {

class ARMTargetMachine;
class FunctionPass;
class MachineCodeEmitter;
class raw_ostream;

// Enums corresponding to ARM condition codes
namespace ARMCC {
  // The CondCodes constants map directly to the 4-bit encoding of the 
  // condition field for predicated instructions. 
  enum CondCodes {
    EQ,
    NE,
    HS,
    LO,
    MI,
    PL,
    VS,
    VC,
    HI,
    LS,
    GE,
    LT,
    GT,
    LE,
    AL
  };
  
  inline static CondCodes getOppositeCondition(CondCodes CC){
    switch (CC) {
    default: assert(0 && "Unknown condition code");
    case EQ: return NE;
    case NE: return EQ;
    case HS: return LO;
    case LO: return HS;
    case MI: return PL;
    case PL: return MI;
    case VS: return VC;
    case VC: return VS;
    case HI: return LS;
    case LS: return HI;
    case GE: return LT;
    case LT: return GE;
    case GT: return LE;
    case LE: return GT;
    }
  }
}

inline static const char *ARMCondCodeToString(ARMCC::CondCodes CC) {
  switch (CC) {
  default: assert(0 && "Unknown condition code");
  case ARMCC::EQ:  return "eq";
  case ARMCC::NE:  return "ne";
  case ARMCC::HS:  return "hs";
  case ARMCC::LO:  return "lo";
  case ARMCC::MI:  return "mi";
  case ARMCC::PL:  return "pl";
  case ARMCC::VS:  return "vs";
  case ARMCC::VC:  return "vc";
  case ARMCC::HI:  return "hi";
  case ARMCC::LS:  return "ls";
  case ARMCC::GE:  return "ge";
  case ARMCC::LT:  return "lt";
  case ARMCC::GT:  return "gt";
  case ARMCC::LE:  return "le";
  case ARMCC::AL:  return "al";
  }
}

FunctionPass *createARMISelDag(ARMTargetMachine &TM);
FunctionPass *createARMCodePrinterPass(raw_ostream &O, ARMTargetMachine &TM);
FunctionPass *createARMCodeEmitterPass(ARMTargetMachine &TM,
                                       MachineCodeEmitter &MCE);
FunctionPass *createARMLoadStoreOptimizationPass();
FunctionPass *createARMConstantIslandPass();

} // end namespace llvm;

// Defines symbolic names for ARM registers.  This defines a mapping from
// register name to register number.
//
#include "ARMGenRegisterNames.inc"

// Defines symbolic names for the ARM instructions.
//
#include "ARMGenInstrNames.inc"


#endif
