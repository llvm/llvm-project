//===-- Sim.h - Top-level interface for Sim representation ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM Sim back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Sim_Sim_H
#define LLVM_LIB_TARGET_Sim_Sim_H

#include "MCTargetDesc/SimMCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class SimTargetMachine;
  class FunctionPass;

} // end namespace llvm;

#endif