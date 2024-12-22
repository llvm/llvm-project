#if LLPC_BUILD_NPI
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
/// to index the operands. Most bundles can be lowered to a single VALU in the
/// AMDGPULowerVGPREncoding pass (with the exception of data movement bundles
/// containing only loads and stores). Replace the V_LOAD/STORE_IDX data
/// operands with staging registers.
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

// OpInLdSt and OpInCoreMI are null if MI is CoreMI, including if V_STORE_IDX is
// the CoreMI
struct BundleItem {
  MachineInstr *MI;
  MachineOperand *OpInLdSt;
  MachineOperand *OpInCoreMI;
  Register StagingReg;
};

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
  const TargetRegisterInfo *TRI;
  const SIInstrInfo *STI;
  bool NeedsAlignedVGPRs;
};

} // End anonymous namespace.

char AMDGPUBundleIdxLdSt::ID = 0;
char &llvm::AMDGPUBundleIdxLdStID = AMDGPUBundleIdxLdSt::ID;

INITIALIZE_PASS(AMDGPUBundleIdxLdSt, DEBUG_TYPE,
                "Bundle indexed load/store with uses", false, false)

bool AMDGPUBundleIdxLdSt::bundleIdxLdSt(MachineInstr *MI) {
  if (MI->isMetaInstruction())
    return false;
  MachineRegisterInfo *MRI = &MI->getParent()->getParent()->getRegInfo();
  MachineBasicBlock *MBB = MI->getParent();
  SmallVector<BundleItem, 4> Worklist;
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

    if (NeedsAlignedVGPRs && MI->getOpcode() != AMDGPU::V_LOAD_IDX &&
        AMDGPU::getRegOperandSize(TRI, MI->getDesc(), Def.getOperandNo()) > 4) {
      // Do not bundle instructions with odd offsets to ensure proper register
      // alignment.
      unsigned Offset =
          STI->getNamedOperand(*StoreMI, AMDGPU::OpName::offset)->getImm();
      if (Offset & 1)
        continue;
    }

    MachineOperand *IdxOp = STI->getNamedOperand(*StoreMI, AMDGPU::OpName::idx);
    IdxList.insert(IdxOp->getReg());
    Worklist.push_back({StoreMI, UseOfMI, &Def, AMDGPU::STG_DSTA});
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
                                  E = Worklist[0].MI->getIterator();
      for (++I; I != E; ++I) {
        if (I->mayStore() && !STI->areMemAccessesTriviallyDisjoint(*MI, *I))
          return false;
      }
    }
  }
  Worklist.push_back({MI, nullptr, nullptr, 0});

  const unsigned NumSrcStagingRegs = 6;
  static const Register StagingRegs[NumSrcStagingRegs] = {
      AMDGPU::STG_SRCA, AMDGPU::STG_SRCB, AMDGPU::STG_SRCC,
      AMDGPU::STG_SRCD, AMDGPU::STG_SRCE, AMDGPU::STG_SRCF};
  unsigned StagingRegIdx = 0;

  for (auto &Use : MI->explicit_uses()) {
    if (StagingRegIdx == NumSrcStagingRegs)
      break;
    if (!Use.isReg())
      continue;
    // TODO-GFX13 Update TwoAddressInstructionPass to handle Bundles
    if (Use.isTied())
      return false;
    Register UseReg = Use.getReg();
    if (!UseReg.isVirtual())
      continue;
    MachineInstr *UseMI = MRI->getVRegDef(UseReg);
    if (!UseMI)
      continue;
    if (UseMI->getOpcode() != AMDGPU::V_LOAD_IDX)
      continue;
    // TODO-GFX13 handle load_idx in different block.
    if (UseMI->getParent() != MBB)
      continue;
    // TODO-GFX13 duplicate V_LOAD_IDX if it is used in multiple insts. Depends
    // on replacement with staging registers
    if (!MRI->hasOneNonDBGUse(UseReg))
      continue;

    if (NeedsAlignedVGPRs &&
        AMDGPU::getRegOperandSize(TRI, MI->getDesc(), Use.getOperandNo()) > 4) {
      // Do not bundle instructions with odd offsets to ensure proper register
      // alignment.
      unsigned Offset =
          STI->getNamedOperand(*UseMI, AMDGPU::OpName::offset)->getImm();
      if (Offset & 1)
        continue;
    }

    // Do not move any V_LOAD_IDX past a V_STORE_IDX because they may alias
    // TODO-GFX13 make this more precise by checking idx and offset
    bool AliasConflict = false;
    MachineBasicBlock::instr_iterator I = UseMI->getIterator(),
                                      E = Worklist[0].MI->getIterator();
    for (++I; I != E; ++I) {
      if (I->getOpcode() == AMDGPU::V_STORE_IDX) {
        AliasConflict = true;
        break;
      }
    }
    if (AliasConflict)
      continue;

    MachineOperand *IdxOp = STI->getNamedOperand(*UseMI, AMDGPU::OpName::idx);
    if (!IdxList.count(IdxOp->getReg())) {
      // TODO-GFX13 Need to implement idx0 saving and restoring in
      // AMDGPUIdxRegAlloc to support 4 unique indexes
      if (IdxList.size() >= 3)
        continue;
      IdxList.insert(IdxOp->getReg());
    }
    Worklist.push_back({UseMI,
                        STI->getNamedOperand(*UseMI, AMDGPU::OpName::data_op),
                        &Use, StagingRegs[StagingRegIdx]});
    StagingRegIdx++;
  }
  if (IdxList.size() == 0)
    return false;

  // Insert bundle where the store was, or where MI was if there was no store.
  auto LastMII = MachineBasicBlock::instr_iterator(Worklist[0].MI);
  auto FirstMII = LastMII;
  // Replace the registers in the bundle with the staging registers
  if (auto *Op = Worklist[0].OpInLdSt) {
    Op->setReg(Worklist[0].StagingReg);
  }
  if (auto *Op = Worklist[0].OpInCoreMI) {
    Op->setReg(Worklist[0].StagingReg);
  }

  for (unsigned I = 1; I < Worklist.size(); I++) {
    MachineInstr *CurMI = Worklist[I].MI;
    CurMI->removeFromParent();
    MBB->insert(FirstMII, CurMI);
    if (auto *Op = Worklist[I].OpInLdSt) {
      Op->setReg(Worklist[I].StagingReg);
    }
    if (auto *Op = Worklist[I].OpInCoreMI) {
      Op->setReg(Worklist[I].StagingReg);
    }
    FirstMII = MachineBasicBlock::instr_iterator(CurMI);
  }
  finalizeBundle(*MBB, FirstMII, ++LastMII);
  return true;
}

bool AMDGPUBundleIdxLdSt::runOnMachineFunction(MachineFunction &MF) {

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  if (!ST.hasVGPRIndexingRegisters())
    return false;

  TRI = ST.getRegisterInfo();
  STI = ST.getInstrInfo();
  NeedsAlignedVGPRs = ST.needsAlignedVGPRs();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (auto &MI : MBB) {
      Changed = bundleIdxLdSt(&MI);
    }
  }
  return Changed;
}
#endif /* LLPC_BUILD_NPI */
