//===- GCNBreakLoadClusterDeps.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Post-RA pass that breaks false (WAR/WAW) anti-dependencies on the address
/// computation feeding clusterable memory loads, so that the downstream post-RA
/// load-clustering scheduler can issue the loads as a burst and expose more
/// memory-level parallelism.
///
/// Register allocation may pack the per-lane index/address computation of
/// several adjacent loads into a small set of registers (e.g. funneling every
/// extracted index through a single scratch VGPR, or reusing one address
/// register pair across two loads). Those reuses are anti-dependencies: they
/// serialize the address chains and pin the loads apart even though the loads
/// are semantically independent. The post-RA MachineScheduler's load-cluster
/// mutation cannot rename registers, so it cannot undo them.
///
/// This pass renames the reused registers on those address chains to free
/// registers scavenged from the function, bounded by the VGPR budget of the
/// current occupancy so it never spends registers that would drop the number of
/// concurrent waves. It performs no rescheduling itself: once the false
/// dependencies are gone the existing load-cluster scheduler does the reorder.
//===----------------------------------------------------------------------===//

#include "GCNBreakLoadClusterDeps.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

#include <algorithm>
#include <bitset>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

using std::bitset;
using std::pair;
using std::reverse;
using std::tie;
using std::unordered_set;
using std::vector;

using namespace llvm;

#define DEBUG_TYPE "amdgpu-break-load-cluster-deps"

namespace {

/// Target-independent-of-pass-manager implementation.
class GCNBreakLoadClusterDepsImpl {
  const GCNSubtarget *ST = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  unsigned OccupancyBudget;

  bitset<AMDGPU::NUM_TARGET_REGS> getVGPR32Lanes(Register Reg) const;
  pair<bitset<AMDGPU::NUM_TARGET_REGS>, bitset<AMDGPU::NUM_TARGET_REGS>>
  getUsesAndDefsFor(MachineInstr &MI) const;
  Register promoteToSuperRegister(MachineInstr &MI, Register SubReg, bool Defs,
                                  bool Uses);
  Register renameRegister(Register FromReg, Register ToReg, Register RenameReg);
  bool
  findReplaceRegisterOperand(MachineInstr &MI, unsigned OpNum,
                             const bitset<AMDGPU::NUM_TARGET_REGS> &BannedRegs,
                             bool MIMustBeKiller = false);
  bool isVGPRLoad(MachineInstr &MI) const {
    return MI.mayLoad() && MI.getOperand(0).isReg() &&
           TRI->isVGPR(*MRI, MI.getOperand(0).getReg());
  }

public:
  bool run(MachineFunction &MF);
  bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
};

/// Legacy-PM wrapper.
class GCNBreakLoadClusterDepsLegacy : public MachineFunctionPass {
public:
  static char ID;

  GCNBreakLoadClusterDepsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Break Load Cluster Dependencies";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

// Append the 32-bit VGPR lanes of physical VGPR `Reg` (any width) to `Lanes`.
bitset<AMDGPU::NUM_TARGET_REGS>
GCNBreakLoadClusterDepsImpl::getVGPR32Lanes(Register Reg) const {
  bitset<AMDGPU::NUM_TARGET_REGS> ToReturn;
  if (!TRI->isVGPR(*MRI, Reg))
    return ToReturn;

  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(Reg);
  unsigned NumLanes = TRI->getRegSizeInBits(*RC).getFixedValue() / 32;
  if (NumLanes <= 1) { // already a VGPR_32
    ToReturn[Reg] = true;
    return ToReturn;
  }
  for (unsigned C = 0; C < NumLanes; ++C)
    ToReturn[TRI->getSubReg(Reg, TRI->getSubRegFromChannel(C))] = true;
  return ToReturn;
}

pair<bitset<AMDGPU::NUM_TARGET_REGS>, bitset<AMDGPU::NUM_TARGET_REGS>>
GCNBreakLoadClusterDepsImpl::getUsesAndDefsFor(MachineInstr &MI) const {
  pair<bitset<AMDGPU::NUM_TARGET_REGS>, bitset<AMDGPU::NUM_TARGET_REGS>>
      ToReturn;
  for (unsigned I = 0; I < MI.getNumExplicitOperands(); I++)
    if (MI.getOperand(I).isReg())
      (*(MI.getOperand(I).isDef() ? &ToReturn.first : &ToReturn.second)) |=
          getVGPR32Lanes(MI.getOperand(I).getReg());
  return ToReturn;
}

Register GCNBreakLoadClusterDepsImpl::promoteToSuperRegister(MachineInstr &MI,
                                                             Register SubReg,
                                                             bool Defs,
                                                             bool Uses) {
  for (MachineOperand &Operand : MI.explicit_operands())
    if (Operand.isReg() && (Defs || Operand.isUse()) &&
        (Uses || Operand.isDef()) &&
        TRI->isSuperRegister(SubReg, Operand.getReg()))
      SubReg = Operand.getReg();

  return SubReg;
}

Register GCNBreakLoadClusterDepsImpl::renameRegister(Register FromReg,
                                                     Register ToReg,
                                                     Register RenameReg) {
  if (RenameReg == FromReg)
    return ToReg;
  if (unsigned Idx =
          TRI->getSubRegIndex(FromReg.asMCReg(), RenameReg.asMCReg()))
    return TRI->getSubReg(ToReg.asMCReg(), Idx);
  return RenameReg;
}

bool GCNBreakLoadClusterDepsImpl::findReplaceRegisterOperand(
    MachineInstr &MI, unsigned OpNum,
    const bitset<AMDGPU::NUM_TARGET_REGS> &BannedRegs, bool MIMustBeKiller) {
  MachineBasicBlock& MBB = *MI.getParent();
  MachineInstr *DefToRename = nullptr, *KillerIns = nullptr;
  Register OldReg = MI.getOperand(OpNum).getReg();
  bitset<AMDGPU::NUM_TARGET_REGS> OldRegClobbers = getVGPR32Lanes(OldReg);
  if (MI.getOperand(OpNum).isDef())
    DefToRename = &MI;
  else
    KillerIns = &MI;

  bool Changed = true;
  while (Changed) {
    Changed = false;

    // First, go forward from def to find the kill
    if (DefToRename) {
      bitset<AMDGPU::NUM_TARGET_REGS> ClobberedSubregs;
      MachineInstr *NewKiller = KillerIns ? KillerIns : nullptr;
      Register OldOldReg = OldReg;
      for (MachineBasicBlock::iterator It =
               std::next(DefToRename->getIterator());
           (OldRegClobbers & ~ClobberedSubregs).any() && It != MBB.end();
           ++It) {
        auto Subregs = getUsesAndDefsFor(*It);
        if ((Subregs.second & OldRegClobbers).any())
          NewKiller = &*It;
        ClobberedSubregs |= Subregs.first;

        //Handle promoting OldReg to a super-register of it
        Register NewOldReg = promoteToSuperRegister(*std::prev(It),OldReg,true,false);
        NewOldReg = promoteToSuperRegister(*It, NewOldReg, false, true);
        if (NewOldReg != OldReg) {
          Changed = true;
          OldReg = NewOldReg;
          OldRegClobbers = getVGPR32Lanes(OldReg);
        }
      }

      if (NewKiller != KillerIns) {
        KillerIns = NewKiller;
        Changed = true;
      }

      // Are we live out with no true kill?  Fail: we can't do this with a
      // block-local analysis.
      if (OldOldReg == OldReg && (OldRegClobbers & ~ClobberedSubregs).any()) {
        LiveRegUnits LRU(*TRI);
        LRU.addLiveOuts(MBB);
        if (!LRU.available(OldReg))
          return false;
      }
    }

    //Second, go backward from killer to find the def
    if (KillerIns) {
      bitset<AMDGPU::NUM_TARGET_REGS> ClobberedSubregs;
      MachineInstr *NewDef = DefToRename ? DefToRename : nullptr;
      for (MachineBasicBlock::reverse_iterator RIt =
               std::next(KillerIns->getReverseIterator());
           RIt != MBB.rend(); ++RIt) {
        if (RIt->modifiesRegister(OldReg, TRI))
          ClobberedSubregs |= getUsesAndDefsFor(*RIt).first;

        if ((OldRegClobbers & ~ClobberedSubregs).none()) {
          NewDef = &*RIt;
          break;
        }
      }

      if (NewDef != DefToRename) {
        DefToRename = NewDef;
        Changed = true;
      }
    }
  }

  if (!DefToRename || !KillerIns || (MIMustBeKiller && KillerIns != &MI))
    return false;

  // Pre-mutation guards over the window [DefToRename, KillerIns].  Bail before
  // changing anything if the rename can't be done correctly.

  // Tied guard (conservative, whole footprint): a tied def/use pair is pinned to
  // a single physical register.  If it couples something we must rename (a def,
  // or a use reading the renamed value) with something we must not (a use whose
  // incoming value is defined before the window), no single register satisfies
  // both and renaming would silently corrupt the untouched side.  We don't
  // distinguish those cases, so bail on any tied operand overlapping OldReg.
  for (MachineBasicBlock::iterator It = DefToRename->getIterator(),
                                   End = std::next(KillerIns->getIterator());
       It != End; ++It)
    for (const MachineOperand &Operand : It->operands())
      if (Operand.isReg() && Operand.isTied() &&
          TRI->regsOverlap(Operand.getReg(), OldReg))
        return false;

  // Now, perform the rename between (DefToRename, KillerIns)

  // Find a free reg
  LiveRegUnits LRU(*TRI);
  LRU.addLiveOuts(MBB);
  for (MachineBasicBlock::reverse_iterator LiveRIt = MBB.rbegin();
       &*LiveRIt != &*KillerIns; ++LiveRIt)
    LRU.stepBackward(*LiveRIt);
  for (MachineBasicBlock::reverse_iterator AccumIt =
           KillerIns->getReverseIterator();
       &*AccumIt != DefToRename; ++AccumIt)
    LRU.accumulate(*AccumIt);
  
  // Iterate over registers in physical register class
  const TargetRegisterClass &DefinedRegClass =
      *TRI->getPhysRegBaseClass(OldReg);
  unsigned I;
  for (I = 0; I < DefinedRegClass.getRegisters().size(); I++) {
    if (TRI->getHWRegIndex(DefinedRegClass.getRegisters()[I]) >=
        OccupancyBudget)
      continue;
    if (LRU.available(DefinedRegClass.getRegisters()[I]) &&
        (getVGPR32Lanes(DefinedRegClass.getRegisters()[I]) & BannedRegs).none())
      break;
  }

  // Fail if we couldn't find a suitable free register
  if (I == DefinedRegClass.getRegisters().size())
    return false;

  auto renameRegisters = [&](bool DryRun) {
    bitset<AMDGPU::NUM_TARGET_REGS> RedefinedRegs;
    // Actually rename the register
    for (unsigned Op = 0; Op < DefToRename->getNumExplicitOperands(); Op++)
      if (DefToRename->getOperand(Op).isReg() &&
          DefToRename->getOperand(Op).isDef() &&
          TRI->regsOverlap(DefToRename->getOperand(Op).getReg(), OldReg)) {
        Register NewDef =
            renameRegister(OldReg, DefinedRegClass.getRegisters()[I],
                           DefToRename->getOperand(Op).getReg());
        if(!DryRun)
          DefToRename->getOperand(Op).setReg(NewDef);
        else if (!DefToRename->getOperand(Op).isRenamable())
          return false;
        RedefinedRegs |= getVGPR32Lanes(NewDef);
      }
    for (MachineBasicBlock::iterator RenameIt =
             std::next(DefToRename->getIterator());
         RenameIt != KillerIns; ++RenameIt)
      for (int Op = RenameIt->getNumExplicitOperands() - 1; Op >= 0; Op--)
        if (RenameIt->getOperand(Op).isReg() &&
            TRI->regsOverlap(RenameIt->getOperand(Op).getReg(), OldReg) &&
            (RenameIt->getOperand(Op).isDef() ||
             (RedefinedRegs & getVGPR32Lanes(renameRegister(
                                  OldReg, DefinedRegClass.getRegisters()[I],
                                  RenameIt->getOperand(Op).getReg())))
                 .any())) {
          Register NewReg =
              renameRegister(OldReg, DefinedRegClass.getRegisters()[I],
                             RenameIt->getOperand(Op).getReg());
          if(!DryRun)
            RenameIt->getOperand(Op).setReg(NewReg);
          else if (!RenameIt->getOperand(Op).isRenamable())
            return false;
          
          if (RenameIt->getOperand(Op).isDef())
            RedefinedRegs |= getVGPR32Lanes(NewReg);
        }
    for (unsigned Op = 0; Op < KillerIns->getNumExplicitOperands(); Op++)
      if (KillerIns->getOperand(Op).isReg() &&
          KillerIns->getOperand(Op).isUse() &&
          TRI->regsOverlap(KillerIns->getOperand(Op).getReg(), OldReg))
        if(!DryRun)
          KillerIns->getOperand(Op).setReg(
              renameRegister(OldReg, DefinedRegClass.getRegisters()[I],
                             KillerIns->getOperand(Op).getReg()));
        else if (!KillerIns->getOperand(Op).isRenamable())
          return false;

    return true;
  };

  if (!renameRegisters(true))
    return false;

  renameRegisters(false);
  return true;
}

bool GCNBreakLoadClusterDepsImpl::runOnMachineBasicBlock(
    MachineBasicBlock &MBB) {
  bool ToReturn = false;

  // Find clusterable loads whose address operands share a register with an
  // earlier load's address, or whose address def chains funnel through a
  // common scratch register (WAR/WAW anti-dependencies).
  vector<MachineInstr *> AllVectorLoads;
  for (MachineInstr &MI : MBB)
    if (isVGPRLoad(MI))
      AllVectorLoads.push_back(&MI);

  reverse(AllVectorLoads.begin(), AllVectorLoads.end()); // efficiency
  bitset<AMDGPU::NUM_TARGET_REGS> UsedLoadSourcePhysregs, UsedLoadDestPhysregs;
  unordered_set<MachineInstr*> ClusterLoads;
  while (!AllVectorLoads.empty()) {
    MachineInstr &VecLoadIns = *AllVectorLoads.back();
    bitset<AMDGPU::NUM_TARGET_REGS> InsDefs, InsUses;
    tie(InsDefs, InsUses) = getUsesAndDefsFor(VecLoadIns);

    if (ClusterLoads.size() && !ClusterLoads.count(&VecLoadIns)) {
      ClusterLoads.clear();
      UsedLoadSourcePhysregs.reset();
      UsedLoadDestPhysregs.reset();
      AllVectorLoads.pop_back();
      continue;
    } else if (!ClusterLoads.count(&VecLoadIns)) {
      bitset<AMDGPU::NUM_TARGET_REGS> ClusterRAWHazards;
      for (MachineBasicBlock::iterator ForwardIt = VecLoadIns.getIterator();
           ForwardIt != MBB.end(); ++ForwardIt) {
        if (isVGPRLoad(*ForwardIt)) {
          bitset<AMDGPU::NUM_TARGET_REGS> UsedVGPRs;
          for (MachineOperand &Operand : ForwardIt->uses())
            if (Operand.isReg() && Operand.isUse() &&
                TRI->isVGPR(*MRI, Operand.getReg()))
              UsedVGPRs |= getVGPR32Lanes(Operand.getReg());
          
          if ((ClusterRAWHazards & UsedVGPRs).any())
            break;

          ClusterLoads.insert(&*ForwardIt);
          ClusterRAWHazards |=
              getVGPR32Lanes(ForwardIt->getOperand(0).getReg());
        } else
          for (MachineOperand &Operand : ForwardIt->defs())
            if (TRI->isVGPR(*MRI,Operand.getReg()))
              ClusterRAWHazards &= getVGPR32Lanes(Operand.getReg());
      }
    } else
      ClusterLoads.erase(&VecLoadIns);

    // If it's used or defined by a load that could be in our cluster, it's
    // _NOT_ free.
    bitset<AMDGPU::NUM_TARGET_REGS> BannedRegs =
      UsedLoadDestPhysregs | UsedLoadSourcePhysregs;
    for (MachineInstr *FutureVecLoad : ClusterLoads) {
      auto UsesAndDefs = getUsesAndDefsFor(*FutureVecLoad);
      BannedRegs |= UsesAndDefs.first;
      BannedRegs |= UsesAndDefs.second;
    }
    
    // Check if we have something to rename due to WAR
    bitset<AMDGPU::NUM_TARGET_REGS> WarConflicts =
      (InsUses | InsDefs) & (UsedLoadSourcePhysregs | UsedLoadDestPhysregs);
    while (WarConflicts.any()) {
      Register OldReg = WarConflicts._Find_first();
      unsigned OpNum;
      for (OpNum = 0; OpNum < VecLoadIns.getNumExplicitOperands(); OpNum++)
        if (VecLoadIns.getOperand(OpNum).isReg() &&
            TRI->regsOverlap(OldReg, VecLoadIns.getOperand(OpNum).getReg()))
          break;
      assert(OpNum != VecLoadIns.getNumExplicitOperands() &&
             "There should be a conflicting register operand.  Where is it?");
      if (!findReplaceRegisterOperand(VecLoadIns, OpNum, BannedRegs))
        break;
      ToReturn = true;
      
      tie(InsDefs, InsUses) = getUsesAndDefsFor(VecLoadIns);
      WarConflicts =
          (InsUses | InsDefs) & (UsedLoadSourcePhysregs | UsedLoadDestPhysregs);
    }
    
    // Check if we have something to rename due to WAR
    bitset<AMDGPU::NUM_TARGET_REGS> SelfConflicts = InsUses & InsDefs;
    while (SelfConflicts.any()) {
      Register OldReg = SelfConflicts._Find_first();
      unsigned OpNum;
      for (OpNum = 0; OpNum < VecLoadIns.getNumExplicitOperands(); OpNum++)
        if (VecLoadIns.getOperand(OpNum).isReg() &&
            VecLoadIns.getOperand(OpNum).isUse() &&
            TRI->regsOverlap(OldReg, VecLoadIns.getOperand(OpNum).getReg()))
          break;
      assert(OpNum != VecLoadIns.getNumExplicitOperands() &&
             "There should be a conflicting register operand.  Where is it?");
      bitset<AMDGPU::NUM_TARGET_REGS> SelfBannedRegs = BannedRegs | InsDefs;
      if (!findReplaceRegisterOperand(VecLoadIns, OpNum, SelfBannedRegs, true))
        break;
      ToReturn = true;
      
      tie(InsDefs, InsUses) = getUsesAndDefsFor(VecLoadIns);
      SelfConflicts = InsUses & InsDefs;
    }
    
    // Coda
    UsedLoadDestPhysregs |= InsDefs;
    UsedLoadSourcePhysregs |= InsUses;
    AllVectorLoads.pop_back();
  }

  // Scavenge free VGPRs and rename along each address def chain so the chains
  // become register-disjoint, staying within the budget.  The downstream
  // post-RA load-cluster scheduler then reorders the now independent loads into
  // a burst.

  return ToReturn;
}

bool GCNBreakLoadClusterDepsImpl::run(MachineFunction &MF) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  TRI = ST->getRegisterInfo();
  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  // getDynamicVGPRBlockSize() already returns 0 when dynamic VGPRs are
  // disabled, so no need to guard on isDynamicVGPREnabled().
  unsigned DynamicBlockSize =
      MF.getInfo<SIMachineFunctionInfo>()->getDynamicVGPRBlockSize();
  OccupancyBudget = ST->getMaxNumVGPRs(
      ST->getOccupancyWithNumVGPRs(
          TRI->getNumUsedPhysRegs(*MRI, AMDGPU::VGPR_32RegClass),
          DynamicBlockSize),
      DynamicBlockSize);

  bool ToReturn = false;
  for (MachineBasicBlock &MBB : MF)
    ToReturn |= runOnMachineBasicBlock(MBB);

  return ToReturn;
}

bool GCNBreakLoadClusterDepsLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  return GCNBreakLoadClusterDepsImpl().run(MF);
}

PreservedAnalyses
GCNBreakLoadClusterDepsPass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  if (!GCNBreakLoadClusterDepsImpl().run(MF))
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

char GCNBreakLoadClusterDepsLegacy::ID = 0;

char &llvm::GCNBreakLoadClusterDepsID = GCNBreakLoadClusterDepsLegacy::ID;

INITIALIZE_PASS(GCNBreakLoadClusterDepsLegacy, DEBUG_TYPE,
                "AMDGPU Break Load Cluster Dependencies", false, false)
