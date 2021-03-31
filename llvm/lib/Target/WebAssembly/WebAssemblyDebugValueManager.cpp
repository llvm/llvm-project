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
#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineInstr.h"

using namespace llvm;

WebAssemblyDebugValueManager::WebAssemblyDebugValueManager(
    MachineInstr *Instr) {
  // This code differs from MachineInstr::collectDebugValues in that it scans
  // the whole BB, not just contiguous DBG_VALUEs.
  if (!Instr->getOperand(0).isReg())
    return;
  CurrentReg = Instr->getOperand(0).getReg();

  MachineBasicBlock::iterator DI = *Instr;
  ++DI;
  for (MachineBasicBlock::iterator DE = Instr->getParent()->end(); DI != DE;
       ++DI) {
    if (DI->isDebugValue() &&
        DI->hasDebugOperandForReg(Instr->getOperand(0).getReg()))
      DbgValues.push_back(&*DI);
  }
}

void WebAssemblyDebugValueManager::move(MachineInstr *Insert) {
  MachineBasicBlock *MBB = Insert->getParent();
  for (MachineInstr *DBI : reverse(DbgValues))
    MBB->splice(Insert, DBI->getParent(), DBI);
}

void WebAssemblyDebugValueManager::updateReg(unsigned Reg) {
  for (auto *DBI : DbgValues)
    for (auto &MO : DBI->getDebugOperandsForReg(CurrentReg))
      MO.setReg(Reg);
  CurrentReg = Reg;
}

void WebAssemblyDebugValueManager::clone(MachineInstr *Insert,
                                         unsigned NewReg) {
  MachineBasicBlock *MBB = Insert->getParent();
  MachineFunction *MF = MBB->getParent();
  for (MachineInstr *DBI : reverse(DbgValues)) {
    MachineInstr *Clone = MF->CloneMachineInstr(DBI);
    for (auto &MO : Clone->getDebugOperandsForReg(CurrentReg))
      MO.setReg(NewReg);
    MBB->insert(Insert, Clone);
  }
}

void WebAssemblyDebugValueManager::replaceWithLocal(unsigned LocalId) {
  for (auto *DBI : DbgValues) {
    auto IndexType = DBI->isIndirectDebugValue()
                         ? llvm::WebAssembly::TI_LOCAL_INDIRECT
                         : llvm::WebAssembly::TI_LOCAL;
    for (auto &MO : DBI->getDebugOperandsForReg(CurrentReg))
      MO.ChangeToTargetIndex(IndexType, LocalId);
  }
}
