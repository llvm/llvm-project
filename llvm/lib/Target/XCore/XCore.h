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

#include "MCTargetDesc/XCoreMCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class FunctionPass;
  class TargetMachine;
  class XCoreTargetMachine;
  class formatted_raw_ostream;

  FunctionPass *createXCoreISelDag(XCoreTargetMachine &TM,
                                   CodeGenOpt::Level OptLevel);

} // end namespace llvm;

#endif
