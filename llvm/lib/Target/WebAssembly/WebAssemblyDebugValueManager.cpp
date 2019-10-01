//===-- WebAssemblyDebugValueManager.cpp - WebAssembly DebugValue Manager -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the manager for MachineInstr DebugValues.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyDebugValueManager.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineInstr.h"

using namespace llvm;

WebAssemblyDebugValueManager::WebAssemblyDebugValueManager(
    MachineInstr *Instr) {
  Instr->collectDebugValues(DbgValues);
}

void WebAssemblyDebugValueManager::move(MachineInstr *Insert) {
  MachineBasicBlock *MBB = Insert->getParent();
  for (MachineInstr *DBI : reverse(DbgValues))
    MBB->splice(Insert, DBI->getParent(), DBI);
}

void WebAssemblyDebugValueManager::updateReg(unsigned Reg) {
  for (auto *DBI : DbgValues)
    DBI->getOperand(0).setReg(Reg);
}

void WebAssemblyDebugValueManager::clone(MachineInstr *Insert,
                                         unsigned NewReg) {
  MachineBasicBlock *MBB = Insert->getParent();
  MachineFunction *MF = MBB->getParent();
  for (MachineInstr *DBI : reverse(DbgValues)) {
    MachineInstr *Clone = MF->CloneMachineInstr(DBI);
    Clone->getOperand(0).setReg(NewReg);
    MBB->insert(Insert, Clone);
  }
}
