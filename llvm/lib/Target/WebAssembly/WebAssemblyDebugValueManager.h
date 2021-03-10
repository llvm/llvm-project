// WebAssemblyDebugValueManager.h - WebAssembly DebugValue Manager -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the WebAssembly-specific
/// manager for DebugValues associated with the specific MachineInstr.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYDEBUGVALUEMANAGER_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYDEBUGVALUEMANAGER_H

#include "llvm/ADT/SmallVector.h"

namespace llvm {

class MachineInstr;

class WebAssemblyDebugValueManager {
  SmallVector<MachineInstr *, 2> DbgValues;
  unsigned CurrentReg;

public:
  WebAssemblyDebugValueManager(MachineInstr *Instr);

  void move(MachineInstr *Insert);
  void updateReg(unsigned Reg);
  void clone(MachineInstr *Insert, unsigned NewReg);
  void replaceWithLocal(unsigned LocalId);
};

} // end namespace llvm

#endif
