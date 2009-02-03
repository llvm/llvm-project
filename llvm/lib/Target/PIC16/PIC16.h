//===-- PIC16.h - Top-level interface for PIC16 representation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in 
// the LLVM PIC16 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_PIC16_H
#define LLVM_TARGET_PIC16_H

#include <iosfwd>
#include <cassert>

namespace llvm {
  class PIC16TargetMachine;
  class FunctionPass;
  class MachineCodeEmitter;
  class raw_ostream;

namespace PIC16CC {
  enum CondCodes {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    ULT,
    UGT,
    ULE,
    UGE
  };
}

  inline static const char *PIC16CondCodeToString(PIC16CC::CondCodes CC) {
    switch (CC) {
    default: assert(0 && "Unknown condition code");
    case PIC16CC::NE:  return "ne";
    case PIC16CC::EQ:   return "eq";
    case PIC16CC::LT:   return "lt";
    case PIC16CC::ULT:   return "lt";
    case PIC16CC::LE:  return "le";
    case PIC16CC::GT:  return "gt";
    case PIC16CC::UGT:  return "gt";
    case PIC16CC::GE:   return "ge";
    }
  }

  inline static bool isSignedComparison(PIC16CC::CondCodes CC) {
    switch (CC) {
    default: assert(0 && "Unknown condition code");
    case PIC16CC::NE:  
    case PIC16CC::EQ: 
    case PIC16CC::LT:
    case PIC16CC::LE:
    case PIC16CC::GE:
    case PIC16CC::GT:
      return true;
    case PIC16CC::ULT:
    case PIC16CC::UGT:
    case PIC16CC::ULE:
    case PIC16CC::UGE:
      return false;   // condition codes for unsigned comparison. 
    }
  }


  FunctionPass *createPIC16ISelDag(PIC16TargetMachine &TM);
  FunctionPass *createPIC16CodePrinterPass(raw_ostream &OS, 
                                           PIC16TargetMachine &TM);
} // end namespace llvm;

// Defines symbolic names for PIC16 registers.  This defines a mapping from
// register name to register number.
#include "PIC16GenRegisterNames.inc"

// Defines symbolic names for the PIC16 instructions.
#include "PIC16GenInstrNames.inc"

#endif
