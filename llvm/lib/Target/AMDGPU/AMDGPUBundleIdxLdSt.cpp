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
  SmallVector<MachineOperand *> OpsInCoreMI;
  Register StagingReg;
  Register OpReg;
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
  void recoverIdx0ForPrivateUse(const MachineRegisterInfo &MRI,
                                SmallVector<BundleItem, 4> &Worklist,
                                std::unordered_set<unsigned> &IdxList,
                                unsigned &SrcStagingRegIdx);
  const TargetRegisterInfo *TRI;
  const SIInstrInfo *STI;
  bool NeedsAlignedVGPRs;
};

} // End anonymous namespace.

char AMDGPUBundleIdxLdSt::ID = 0;
char &llvm::AMDGPUBundleIdxLdStID = AMDGPUBundleIdxLdSt::ID;

INITIALIZE_PASS(AMDGPUBundleIdxLdSt, DEBUG_TYPE,
                "Bundle indexed load/store with uses", false, false)

constexpr unsigned NumSrcStagingRegs = 6;

void AMDGPUBundleIdxLdSt::recoverIdx0ForPrivateUse(
    const MachineRegisterInfo &MRI, SmallVector<BundleItem, 4> &Worklist,
    std::unordered_set<unsigned> &IdxList, unsigned &SrcStagingRegIdx) {
  // First, find the idx reg with the least V_LOAD_IDX uses
  // Second, remove the loads that use the idx from the worklist
  // and remap the staging regs to get an updated SrcStagingRegIdx
  DenseMap<unsigned, unsigned> IdxRegUseCounts(NumSrcStagingRegs);
  assert(Worklist.size() >= 4 &&
         "Shouldn't be attempting to recover idx0 if there aren't at least 4 "
         "bundled instructions");
  static_assert(AMDGPU::STG_DSTA < AMDGPU::STG_SRCA &&
                "idx0 staging reg recovery is incorrect if staging reg "
                "ordering is changed");
  for (auto &BI : Worklist) {
    // Only consider src staging registers for implicit use of idx0, to simplify
    // the algorithm
    if (BI.StagingReg < AMDGPU::STG_SRCA)
      continue;
    Register IdxOpReg =
        STI->getNamedOperand(*BI.MI, AMDGPU::OpName::idx)->getReg();
    if (IdxRegUseCounts.find(IdxOpReg) == IdxRegUseCounts.end())
      IdxRegUseCounts[IdxOpReg] = 0;
    IdxRegUseCounts[IdxOpReg]++;
  }
  Register MinUseIdxOpReg;
  unsigned MinUses = std::numeric_limits<unsigned>::max();
  for (const auto &IdxRegUseCount : IdxRegUseCounts) {
    if (IdxRegUseCount.second < MinUses) {
      MinUseIdxOpReg = IdxRegUseCount.first;
      MinUses = IdxRegUseCount.second;
    }
  }
  assert(MinUseIdxOpReg.isValid() &&
         "There should always be at least one staging register with only one "
         "use, otherwise we wouldn't have to recover idx0");
  IdxList.erase(MinUseIdxOpReg);
  unsigned NewSrcStagingRegIdx = 0;
  constexpr unsigned DefaultValueSentinel = NumSrcStagingRegs;
  IndexedMap<unsigned> NewRegMap(DefaultValueSentinel);
  NewRegMap.resize(NumSrcStagingRegs);
  // remaps multiple uses of the same staging reg to the same new staging reg,
  // to preserve sequential usage of staging regs
  llvm::erase_if(Worklist, [&](auto &BI) {
    if (BI.StagingReg < AMDGPU::STG_SRCA) {
      return false;
    }
    if (STI->getNamedOperand(*BI.MI, AMDGPU::OpName::idx)->getReg() !=
        MinUseIdxOpReg) {
      unsigned I = BI.StagingReg - AMDGPU::STG_SRCA;
      if (NewRegMap[I] == DefaultValueSentinel) {
        NewRegMap[I] = AMDGPU::STG_SRCA + NewSrcStagingRegIdx;
        NewSrcStagingRegIdx++;
      }
      BI.StagingReg = NewRegMap[I];
      return false;
    }
    for (auto *CoreMIOp : BI.OpsInCoreMI) {
      CoreMIOp->setReg(BI.OpReg);
    }
    return true;
  });
  SrcStagingRegIdx = NewSrcStagingRegIdx;
}

bool AMDGPUBundleIdxLdSt::bundleIdxLdSt(MachineInstr *MI) {
  if (MI->isMetaInstruction())
    return false;
  // Prevent cycles in data-flow from multiple defs. This check is too coarse.
  // We could fix this with per BB analysis, but prefer to fix it later while
  // extending the algorithm to multiple BBs.
  if (MI->getNumExplicitDefs() > 1)
    return false;
  // TODO-GFX13 Update TwoAddressInstructionPass to handle Bundles
  if (MI->isConvertibleTo3Addr() || MI->isRegSequence() || MI->isInsertSubreg())
    return false;
  // COPY would be lowered to v_mov, which is equivalent to not bundling at all,
  // and further optimization of the COPY would be blocked by the BUNDLE, so
  // skip it.
  if (MI->isCopy())
    return false;

  MachineFunction *MF = MI->getParent()->getParent();
  MachineRegisterInfo *MRI = &MF->getRegInfo();
  MachineBasicBlock *MBB = MI->getParent();
  SmallVector<BundleItem, 4> Worklist;
  std::unordered_set<unsigned> IdxList;
  bool UsesIdx0ForPrivate = false;
  bool UsesIdx0ForDynamic = false;

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
    if (UseOfMI->getSubReg() != 0)
      continue;
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
    Worklist.push_back({StoreMI, UseOfMI, {&Def}, AMDGPU::STG_DSTA, DefReg});
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
  Worklist.push_back({MI, nullptr, {}, 0, 0});

  static const Register StagingRegs[NumSrcStagingRegs] = {
      AMDGPU::STG_SRCA, AMDGPU::STG_SRCB, AMDGPU::STG_SRCC,
      AMDGPU::STG_SRCD, AMDGPU::STG_SRCE, AMDGPU::STG_SRCF};
  unsigned StagingRegIdx = 0;
  for (auto &Use : MI->explicit_uses()) {
    if (StagingRegIdx == NumSrcStagingRegs)
      break;
    if (!Use.isReg())
      continue;
    if (Use.getSubReg() != 0)
      continue;
    // TODO-GFX13 Update TwoAddressInstructionPass to handle Bundles
    if (Use.isTied())
      return false;
    Register UseReg = Use.getReg();
    if (!UseReg.isVirtual())
      continue;
    MachineInstr *LoadMI = MRI->getVRegDef(UseReg);
    if (!LoadMI)
      continue;
    if (LoadMI->getOpcode() != AMDGPU::V_LOAD_IDX) {
      if (UsesIdx0ForPrivate)
        continue;
      // Check if a reg use needs a private VGPR of any kind
      const TargetRegisterClass *RegClass = MRI->getRegClass(UseReg);
      if (TRI->getCommonSubClass(RegClass, &AMDGPU::VGPR_32RegClass)) {
        if (UsesIdx0ForDynamic)
          recoverIdx0ForPrivateUse(*MRI, Worklist, IdxList, StagingRegIdx);
        UsesIdx0ForPrivate = true;
        UsesIdx0ForDynamic = false;
      }
      continue;
    }
    // TODO-GFX13 handle load_idx in different block.
    if (LoadMI->getParent() != MBB)
      continue;

    if (NeedsAlignedVGPRs &&
        AMDGPU::getRegOperandSize(TRI, MI->getDesc(), Use.getOperandNo()) > 4) {
      // Do not bundle instructions with odd offsets to ensure proper register
      // alignment.
      unsigned Offset =
          STI->getNamedOperand(*LoadMI, AMDGPU::OpName::offset)->getImm();
      if (Offset & 1)
        continue;
    }

    // Do not move any V_LOAD_IDX past a V_STORE_IDX because they may alias
    // TODO-GFX13 make this more precise by checking idx and offset
    bool AliasConflict = false;
    MachineBasicBlock::instr_iterator I = LoadMI->getIterator(),
                                      E = Worklist[0].MI->getIterator();
    for (++I; I != E; ++I) {
      if (I->getOpcode() == AMDGPU::V_STORE_IDX) {
        AliasConflict = true;
        break;
      }
    }
    if (AliasConflict)
      continue;

    MachineOperand *IdxOp = STI->getNamedOperand(*LoadMI, AMDGPU::OpName::idx);
    if (!IdxList.count(IdxOp->getReg())) {
      // If a bundle would use more than 4 indexes, or if a bundle is
      // using idx0 already through a private vgpr Op, then it can't use idx0
      if (IdxList.size() == 3 && !UsesIdx0ForPrivate) {
        UsesIdx0ForDynamic = true;
      } else if (IdxList.size() == 3 && UsesIdx0ForPrivate) {
        continue;
      } else if (IdxList.size() == 4) {
        recoverIdx0ForPrivateUse(*MRI, Worklist, IdxList, StagingRegIdx);
        UsesIdx0ForDynamic = false;
        UsesIdx0ForPrivate = true;
        continue;
      }
      IdxList.insert(IdxOp->getReg());
    }

    // Duplicate V_LOAD_IDX with uses in multiple instructions.
    auto It = MRI->use_instr_nodbg_begin(UseReg);
    if (++It != MRI->use_instr_nodbg_end()) {
      MachineInstr *DupLoad = MF->CloneMachineInstr(LoadMI);
      MBB->insert(LoadMI, DupLoad);
      LoadMI = DupLoad;
    }

    // Add uses of LoadMI in MI to be replaced.
    // Prevent duplicating loads for multiple uses in one MI. The following
    // iterations of the enclosing loop over MI's uses of the same register will
    // be skipped.
    SmallVector<MachineOperand *> LoadUsesInMI;
    for (auto &Use : make_early_inc_range(MRI->use_operands(UseReg))) {
      if (Use.getParent() == MI) {
        Use.setReg(Register());
        LoadUsesInMI.push_back(&Use);
      }
    }

    Worklist.push_back({LoadMI,
                        STI->getNamedOperand(*LoadMI, AMDGPU::OpName::data_op),
                        LoadUsesInMI, StagingRegs[StagingRegIdx], UseReg});

    StagingRegIdx++;
  }
  if (IdxList.size() == 0)
    return false;

  // Replace the registers in the bundle with the staging registers.

  // Insert bundle where the store was, or where MI was if there was no store.
  auto LastMII = MachineBasicBlock::instr_iterator(Worklist[0].MI);
  auto FirstMII = LastMII;
  if (auto *Op = Worklist[0].OpInLdSt)
    Op->setReg(Worklist[0].StagingReg);
  for (auto *Op : Worklist[0].OpsInCoreMI)
    Op->setReg(Worklist[0].StagingReg);
  for (unsigned I = 1; I < Worklist.size(); I++) {
    MachineInstr *CurMI = Worklist[I].MI;
    CurMI->removeFromParent();
    MBB->insert(FirstMII, CurMI);
    if (auto *Op = Worklist[I].OpInLdSt)
      Op->setReg(Worklist[I].StagingReg);
    for (auto *Op : Worklist[I].OpsInCoreMI)
      Op->setReg(Worklist[I].StagingReg);
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
      Changed |= bundleIdxLdSt(&MI);
    }
  }
  return Changed;
}
