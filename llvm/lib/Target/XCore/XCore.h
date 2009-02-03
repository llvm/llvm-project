//===-- XCore.h - Top-level interface for XCore representation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// XCore back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_XCORE_H
#define TARGET_XCORE_H

namespace llvm {
  class FunctionPass;
  class TargetMachine;
  class XCoreTargetMachine;
  class raw_ostream;

  FunctionPass *createXCoreISelDag(XCoreTargetMachine &TM);
  FunctionPass *createXCoreCodePrinterPass(raw_ostream &OS,
                                           XCoreTargetMachine &TM);
} // end namespace llvm;

// Defines symbolic names for XCore registers.  This defines a mapping from
// register name to register number.
//
#include "XCoreGenRegisterNames.inc"

// Defines symbolic names for the XCore instructions.
//
#include "XCoreGenInstrNames.inc"

#endif
