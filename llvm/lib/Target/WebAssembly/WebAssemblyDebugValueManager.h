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
/// This pass currently does not handle DBG_VALUE_LISTs; they are assumed to
/// have been set to undef in NullifyDebugValueLists pass.
/// TODO Handle DBG_VALUE_LIST
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYDEBUGVALUEMANAGER_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYDEBUGVALUEMANAGER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Register.h"

namespace llvm {

class MachineInstr;

class WebAssemblyDebugValueManager {
  SmallVector<MachineInstr *, 1> DbgValues;
  Register CurrentReg;

public:
  WebAssemblyDebugValueManager(MachineInstr *Def);

  void move(MachineInstr *Insert);
  void clone(MachineInstr *Insert, Register NewReg);
  // Update the register for Def and DBG_VALUEs.
  void updateReg(Register Reg);
  // Replace the current register in DBG_VALUEs with the given LocalId target
  // index.
  void replaceWithLocal(unsigned LocalId);
};

} // end namespace llvm

#endif
