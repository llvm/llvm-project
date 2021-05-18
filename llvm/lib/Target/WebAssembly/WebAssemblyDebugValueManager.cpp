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
  const auto *MF = Instr->getParent()->getParent();
  const auto *TII = MF->getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  // This code differs from MachineInstr::collectDebugValues in that it scans
  // the whole BB, not just contiguous DBG_VALUEs.
  if (!Instr->getOperand(0).isReg())
    return;
  CurrentReg = Instr->getOperand(0).getReg();

  SmallVector<MachineInstr *, 2> DbgValueLists;
  MachineBasicBlock::iterator DI = *Instr;
  ++DI;
  for (MachineBasicBlock::iterator DE = Instr->getParent()->end(); DI != DE;
       ++DI) {
    if (DI->isDebugValue() &&
        DI->hasDebugOperandForReg(Instr->getOperand(0).getReg()))
      DI->getOpcode() == TargetOpcode::DBG_VALUE
          ? DbgValues.push_back(&*DI)
          : DbgValueLists.push_back(&*DI);
  }

  // This class currently cannot handle DBG_VALUE_LISTs correctly. So this
  // converts DBG_VALUE_LISTs to "DBG_VALUE $noreg", which will appear as
  // "optimized out". This can invalidate existing iterators pointing to
  // instructions within this BB from the caller.
  // See https://bugs.llvm.org/show_bug.cgi?id=50361
  // TODO Correctly handle DBG_VALUE_LISTs
  for (auto *DVL : DbgValueLists) {
    BuildMI(*DVL->getParent(), DVL, DVL->getDebugLoc(),
            TII->get(TargetOpcode::DBG_VALUE), false, Register(),
            DVL->getOperand(0).getMetadata(), DVL->getOperand(1).getMetadata());
    DVL->eraseFromParent();
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
