//===- AMDGPUBundleIdxLdSt.cpp - Bundle indexed load/store with uses    ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Form Bundles with VALU instructions and the V_LOAD/STORE_IDX that are used
/// to index the operands. The Bundles can be lowered to a single VALU in the
/// AMDGPULowerVGPREncoding pass.
///
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUResourceUsageAnalysis.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/InitializePasses.h"
#include <unordered_set>

using namespace llvm;

#define DEBUG_TYPE "bundle-indexed-load-store"

namespace {

class AMDGPUBundleIdxLdSt : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUBundleIdxLdSt() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "Bundle indexed load/store with uses";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  bool bundleIdxLdSt(MachineInstr *MI);
  const SIInstrInfo *STI;
};

} // End anonymous namespace.

char AMDGPUBundleIdxLdSt::ID = 0;
char &llvm::AMDGPUBundleIdxLdStID = AMDGPUBundleIdxLdSt::ID;

INITIALIZE_PASS(AMDGPUBundleIdxLdSt, DEBUG_TYPE,
                "Bundle indexed load/store with uses", false, false)

bool AMDGPUBundleIdxLdSt::bundleIdxLdSt(MachineInstr *MI) {
  MachineRegisterInfo *MRI = &MI->getParent()->getParent()->getRegInfo();
  MachineBasicBlock *MBB = MI->getParent();
  SmallVector<MachineInstr *, 4> Worklist;
  std::unordered_set<unsigned> IdxList;
  // Prevent cycles in data-flow from multiple defs. This check is too coarse.
  // We could fix this with per BB analysis, but prefer to fix it later while
  // extending the algorithm to multiple BBs.
  if (MI->getNumExplicitDefs() > 1)
    return false;
  // TODO-GFX13 Update TwoAddressInstructionPass to handle Bundles
  if (MI->isConvertibleTo3Addr() || MI->isRegSequence() || MI->isInsertSubreg())
    return false;
  for (auto &Def : MI->defs()) {
    if (!Def.isReg())
      continue;
    // TODO-GFX13 Update TwoAddressInstructionPass to handle Bundles
    if (Def.isTied())
      return false;
    Register DefReg = Def.getReg();
    if (!MRI->hasOneNonDBGUse(DefReg))
      continue;
    MachineOperand *UseOfMI = &*MRI->use_nodbg_begin(DefReg);
    MachineInstr *StoreMI = UseOfMI->getParent();
    if (StoreMI->getOpcode() != AMDGPU::V_STORE_IDX)
      continue;
    // TODO-GFX13 handle store_idx in different block.
    if (StoreMI->getParent() != MBB)
      continue;
    if (STI->getNamedOperand(*StoreMI, AMDGPU::OpName::data_op)->getReg() !=
        DefReg)
      continue;
    MachineOperand *IdxOp = STI->getNamedOperand(*StoreMI, AMDGPU::OpName::idx);
    IdxList.insert(IdxOp->getReg());
    Worklist.push_back(StoreMI);
  }

  // Check for constraints on moving MI down to StoreMI
  // If MI must happen before I, then we cannot form the bundle by moving
  // MI after I.
  // TODO-GFX13 make this more precise
  if (Worklist.size() > 0) {
    bool MILoads = MI->mayLoad();
    assert(!MI->mayStore() || MILoads &&
                                  "Unexpected MI which produces a values and "
                                  "stores but does not load");
    if (MILoads) {
      MachineBasicBlock::iterator I = MI->getIterator(),
                                  E = Worklist[0]->getIterator();
      for (++I; I != E; ++I) {
        if (I->mayStore() && !STI->areMemAccessesTriviallyDisjoint(*MI, *I))
          return false;
      }
    }
  }
  Worklist.push_back(MI);

  for (const auto &Use : MI->explicit_uses()) {
    if (!Use.isReg())
      continue;
    // TODO-GFX13 Update TwoAddressInstructionPass to handle Bundles
    if (Use.isTied())
      return false;
    Register UseReg = Use.getReg();
    if (!UseReg.isVirtual())
      continue;
    MachineInstr *UseMI = MRI->getVRegDef(UseReg);
    if (UseMI->getOpcode() != AMDGPU::V_LOAD_IDX)
      continue;
    // TODO-GFX13 handle load_idx in different block.
    if (UseMI->getParent() != MBB)
      continue;
    // TODO-GFX13 duplicate V_LOAD_IDX if it is used in multiple insts. Depends
    // on replacement with staging registers
    if (!MRI->hasOneNonDBGUse(UseReg))
      continue;
    MachineOperand *IdxOp = STI->getNamedOperand(*UseMI, AMDGPU::OpName::idx);

    // Do not move any V_LOAD_IDX past a V_STORE_IDX because they may alias
    // TODO-GFX13 make this more precise by checking idx and offset
    bool AliasConflict = false;
    MachineBasicBlock::instr_iterator I = UseMI->getIterator(),
                                      E = Worklist[0]->getIterator();
    for (++I; I != E; ++I) {
      if (I->getOpcode() == AMDGPU::V_STORE_IDX) {
        AliasConflict = true;
        break;
      }
    }
    if (AliasConflict)
      continue;

    if (!IdxList.count(IdxOp->getReg())) {
      // TODO-GFX13 Need to implement idx0 saving and restoring in
      // AMDGPUIdxRegAlloc to support 4 unique indexes
      if (IdxList.size() >= 3)
        continue;
      IdxList.insert(IdxOp->getReg());
    }
    Worklist.push_back(UseMI);
  }
  if (IdxList.size() == 0)
    return false;

  // Insert bundle where the store was, or where MI was if there was no store.
  auto LastMII = MachineBasicBlock::instr_iterator(Worklist[0]);
  auto FirstMII = LastMII;

  for (unsigned I = 1; I < Worklist.size(); I++) {
    Worklist[I]->removeFromParent();
    MBB->insert(FirstMII, Worklist[I]);
    FirstMII = MachineBasicBlock::instr_iterator(Worklist[I]);
  }
  finalizeBundle(*MBB, FirstMII, ++LastMII);
  return true;
}

bool AMDGPUBundleIdxLdSt::runOnMachineFunction(MachineFunction &MF) {

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  STI = ST.getInstrInfo();

  if (!ST.hasVGPRIndexingRegisters())
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto &MI : MBB) {
      Changed = bundleIdxLdSt(&MI);
    }
  }
  return Changed;
}
