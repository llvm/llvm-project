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
/// to index the operands. If the V_LOAD_IDX or VALU instruction are in a
/// different basic block, try to sink them to the their uses so that we are
/// able to form bundles (this pre-bundling sinking phase adapts some of the
/// methods from the generic MachineSink phase). Most bundles can be lowered to
/// a single VALU in the AMDGPULowerVGPREncoding pass (with the exception of
/// data movement bundles containing only loads and stores). Replace the
/// V_LOAD/STORE_IDX data operands with staging registers.
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

constexpr unsigned NumSrcStagingRegs = 6;

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
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<MachineCycleInfoWrapperPass>();
    AU.addPreserved<MachineCycleInfoWrapperPass>();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  bool bundleIdxLdSt(MachineInstr *MI);
  bool sinkInstruction(MachineInstr &MI, bool &SawStore);
  bool sinkLoadsAndCoreMIs(MachineFunction &MF);
  SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>, 4>
  findSuccsToSinkTo(MachineInstr &MI, MachineBasicBlock *MBB);
  void recoverIdx0ForPrivateUse(SmallVector<BundleItem, 4> &Worklist,
                                std::unordered_set<unsigned> &IdxList,
                                unsigned &SrcStagingRegIdx);
  bool hasConflictBetween(MachineBasicBlock *From, MachineBasicBlock *To,
                          MachineInstr &MI);
  bool blockPrologueInterferes(const MachineBasicBlock *BB,
                               MachineBasicBlock::const_iterator End,
                               const MachineInstr &MI);
  void findAllPaths(MachineBasicBlock *Start, MachineBasicBlock *End,
                    SmallVector<SmallVector<MachineBasicBlock *, 8>, 8> &Paths,
                    SmallVector<MachineBasicBlock *, 8> &CurrentPath,
                    DenseSet<MachineBasicBlock *> &Visited);
  SmallVector<SmallVector<MachineBasicBlock *, 8>, 8>
  getAllPathsBetweenBlocks(MachineBasicBlock *Start, MachineBasicBlock *End);

  DenseSet<Register> RegsToClearKillFlags;

  DenseMap<std::pair<MachineBasicBlock *, MachineBasicBlock *>,
           SmallVector<MachineInstr *>>
      ConflictInstrCache;

  DenseMap<std::pair<MachineBasicBlock *, MachineBasicBlock *>, bool>
      HasConflictCache;

  DenseMap<std::pair<MachineBasicBlock *, MachineBasicBlock *>,
           SmallVector<SmallVector<MachineBasicBlock *, 8>, 8>>
      PathsCache;

  const TargetRegisterInfo *TRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  const SIInstrInfo *STI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  AliasAnalysis *AA = nullptr;
  MachineCycleInfo *CI = nullptr;

  bool NeedsAlignedVGPRs;
};

bool sideEffectConflict(MachineInstr &MIa, MachineInstr &MIb) {
  return MIa.hasUnmodeledSideEffects() && MIb.hasUnmodeledSideEffects();
}

// Sink an instruction MI to it's position InsertPos in SuccToSinkTo.
void performSink(MachineInstr &MI, MachineBasicBlock &SuccToSinkTo,
                 MachineBasicBlock::iterator InsertPos) {
  // If we cannot find a location to use (merge with), then we erase the debug
  // location to prevent debug-info driven tools from potentially reporting
  // wrong location information.
  if (!SuccToSinkTo.empty() && InsertPos != SuccToSinkTo.end())
    MI.setDebugLoc(DILocation::getMergedLocation(MI.getDebugLoc(),
                                                 InsertPos->getDebugLoc()));
  else
    MI.setDebugLoc(DebugLoc());

  // Move the instruction.
  MachineBasicBlock *ParentBlock = MI.getParent();
  SuccToSinkTo.splice(InsertPos, ParentBlock, MI,
                      ++MachineBasicBlock::iterator(MI));
}
} // End anonymous namespace.

// Return true if a target defined block prologue instruction interferes
// with a sink candidate.
bool AMDGPUBundleIdxLdSt::blockPrologueInterferes(
    const MachineBasicBlock *BB, MachineBasicBlock::const_iterator End,
    const MachineInstr &MI) {
  for (MachineBasicBlock::const_iterator PI = BB->getFirstNonPHI(); PI != End;
       ++PI) {
    // Only check target defined prologue instructions
    if (!TII->isBasicBlockPrologue(*PI))
      continue;
    for (auto &MO : MI.operands()) {
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      if (!Reg)
        continue;
      if (MO.isUse()) {
        if (Reg.isPhysical() &&
            (TII->isIgnorableUse(MO) || (MRI && MRI->isConstantPhysReg(Reg))))
          continue;
        if (PI->modifiesRegister(Reg, TRI))
          return true;
      } else {
        if (PI->readsRegister(Reg, TRI))
          return true;
        // Check for interference with non-dead defs
        auto *DefOp = PI->findRegisterDefOperand(Reg, TRI, false, true);
        if (DefOp && !DefOp->isDead())
          return true;
      }
    }
  }
  return false;
}

void AMDGPUBundleIdxLdSt::recoverIdx0ForPrivateUse(
    SmallVector<BundleItem, 4> &Worklist, std::unordered_set<unsigned> &IdxList,
    unsigned &SrcStagingRegIdx) {
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

// Find all paths between a given Start and End block.
void AMDGPUBundleIdxLdSt::findAllPaths(
    MachineBasicBlock *Start, MachineBasicBlock *End,
    SmallVector<SmallVector<MachineBasicBlock *, 8>, 8> &Paths,
    SmallVector<MachineBasicBlock *, 8> &CurrentPath,
    DenseSet<MachineBasicBlock *> &Visited) {
  if (Start == End) {
    Paths.push_back(CurrentPath);
    return;
  }

  Visited.insert(Start);
  for (MachineBasicBlock *Succ : Start->successors()) {
    if (Visited.count(Succ) == 0) { // Avoid loops.
      CurrentPath.push_back(Succ);
      findAllPaths(Succ, End, Paths, CurrentPath, Visited);
      CurrentPath.pop_back();
    }
  }
  Visited.erase(Start);
}

// Wraps the recursion and uses a cache for already seen Start/End pairs
SmallVector<SmallVector<MachineBasicBlock *, 8>, 8>
AMDGPUBundleIdxLdSt::getAllPathsBetweenBlocks(MachineBasicBlock *Start,
                                              MachineBasicBlock *End) {

  // Check cache to see if we've already computed these paths.
  auto BlockPair = std::make_pair(Start, End);
  if (auto It = PathsCache.find(BlockPair); It != PathsCache.end())
    return It->second;

  SmallVector<SmallVector<MachineBasicBlock *, 8>, 8> Paths;
  SmallVector<MachineBasicBlock *, 8> CurrentPath;
  DenseSet<MachineBasicBlock *> Visited;
  CurrentPath.push_back(Start);
  findAllPaths(Start, End, Paths, CurrentPath, Visited);

  PathsCache[BlockPair] = Paths;

  return Paths;
}

// Find successors to sink this instruction to, and their insertion points.
// This function uses an all-or-nothing strategy: if we can't sink
// to all basic blocks that have a use, then don't sink at all.
SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>, 4>
AMDGPUBundleIdxLdSt::findSuccsToSinkTo(MachineInstr &MI,
                                       MachineBasicBlock *MBB) {

  SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>, 4>
      Candidates;
  bool IsCoreMI = false;
  bool IsLoadMI = MI.getOpcode() == AMDGPU::V_LOAD_IDX;

  // Loop over all the Defs of the instr, and collect the candidates to sink to.
  size_t TotalUses = 0;
  for (auto &Def : MI.defs()) {
    if (!Def.isReg() || Def.getReg() == 0)
      continue;
    Register DefReg = Def.getReg();

    for (auto U = MRI->use_begin(DefReg); U != MRI->use_end(); U++) {
      assert(U->isReg() && "Expected Use to be reg if Def was reg.");
      TotalUses++;
      MachineInstr *UseMI = U->getParent();
      MachineBasicBlock *UseMBB = UseMI->getParent();

      // If there's a meta/debug use, we wouldn't be able to bundle all uses.
      if (UseMI->isMetaInstruction() || UseMI->isCopy() ||
          UseMI->isDebugOrPseudoInstr() || UseMI->isFakeUse())
        return {};
      // TODO-GFX13 Update TwoAddressInstructionPass to handle Bundles
      if (UseMI->isRegSequence() || UseMI->isInsertSubreg())
        return {};
      // TODO-GFX13 Handle phis.
      if (UseMI->isPHI())
        return {};

      // Determine if this is CoreMI.
      if (!IsLoadMI && UseMI->getOpcode() == AMDGPU::V_STORE_IDX &&
          STI->getNamedOperand(*UseMI, AMDGPU::OpName::data_op)->getReg() ==
              DefReg)
        IsCoreMI = true;
      assert(!(IsCoreMI && IsLoadMI) &&
             "MI can't be both a CoreMI and V_LOAD_IDX.");
      if (!IsLoadMI && !IsCoreMI)
        return {};

      // Check safety of sinking MI to U.
      bool Conflict =
          MI.mayLoad() ? hasConflictBetween(MI.getParent(), UseMBB, MI) : false;
      if (!MI.isSafeToMove(Conflict))
        return {};
      if (!TII->isSafeToSink(MI, UseMBB, CI))
        return {};

      // If the instruction to move defines a dead physical register which is
      // live when leaving the basic block, don't move it because it could turn
      // into a "zombie" define of that phys reg.
      for (const MachineOperand &MO : MI.all_defs()) {
        Register Reg = MO.getReg();
        if (Reg == 0 || !Reg.isPhysical())
          continue;
        if (UseMBB->isLiveIn(Reg))
          return {};
      }

      // Don't move a CoreMI into a cycle.
      if (IsCoreMI && CI->getCycleDepth(UseMBB) > CI->getCycleDepth(MBB)) {
        LLVM_DEBUG(dbgs() << " *** CoreMI sinking to larger cycle depth is "
                             "not profitable\n");
        return {};
      }

      // Determine where to insert into. Skip phi nodes.
      MachineBasicBlock::iterator InsertPos =
          UseMBB->SkipPHIsAndLabels(UseMBB->begin());
      if (blockPrologueInterferes(UseMBB, InsertPos, MI)) {
        LLVM_DEBUG(dbgs() << " *** Not sinking: prologue interference\n");
        return {};
      }

      auto Item = std::make_pair(UseMBB, InsertPos);
      Candidates.push_back(Item);

      // Duplicating CoreMI won't generally be profitable.
      if (IsCoreMI && TotalUses > 1) {
        LLVM_DEBUG(dbgs() << " *** CoreMI has multiple uses; duplicating isn't "
                             "profitable.\n");
        return {};
      }
    }
  }

  return Candidates;
}

// Check if any instruction conflicts with MI between From and To, where a
// conflict is defined as either an alias conflict or both having unmodeled side
// effects. Two caches are used. HasConflictCache is a coarse cache which
// returns true if the pair contains some case we want to treat conservatively
// for all MI (eg. a function call), and returns false if there are no stores at
// all. ConflictInstrCache is used to cache and check the potentially
// conflicting instructions against MI.
bool AMDGPUBundleIdxLdSt::hasConflictBetween(MachineBasicBlock *From,
                                             MachineBasicBlock *To,
                                             MachineInstr &MI) {

  auto BlockPair = std::make_pair(From, To);

  if (auto It = HasConflictCache.find(BlockPair); It != HasConflictCache.end())
    return It->second;

  if (auto It = ConflictInstrCache.find(BlockPair);
      It != ConflictInstrCache.end())
    return llvm::any_of(It->second, [&](MachineInstr *I) {
      bool MayAlias = I->mayAlias(AA, MI, false);
      LLVM_DEBUG(if (MayAlias) {
        dbgs() << " *** Alias conflict with ";
        I->print(dbgs());
      });
      bool SideEffectHazard =
          MI.hasUnmodeledSideEffects() && I->hasUnmodeledSideEffects();
      LLVM_DEBUG(if (SideEffectHazard) {
        dbgs() << " *** Side effect hazard with ";
        I->print(dbgs());
      });
      return SideEffectHazard || MayAlias;
    });

  unsigned int MaxBasicBlockSize = 2000;
  unsigned int MaxPaths = 20;
  unsigned int MaxPathLength = 20;
  bool SawPotentialConflict = false;
  bool HasConflict = false;
  DenseSet<MachineBasicBlock *> HandledBlocks;

  SmallVector<SmallVector<MachineBasicBlock *, 8>, 8> AllPaths =
      getAllPathsBetweenBlocks(From, To);

  // If there are too many paths, treat conservatively to save compile time.
  if (AllPaths.size() > MaxPaths) {
    HasConflictCache[BlockPair] = true;
    return true;
  }

  // Go through all reachable blocks from From.
  for (auto Path : AllPaths) {
    // If any given path is too long, save compiling time.
    if (Path.size() > MaxPathLength) {
      HasConflictCache[BlockPair] = true;
      return true;
    }
    for (auto BB : Path) {
      // We insert the instruction at the start of block To, so no need to
      // worry about conflicts inside To. Conflicts in block From should be
      // already considered when just enter function sinkInstruction.
      if (BB == To || BB == From)
        continue;

      // We already handle this BB in previous iteration.
      if (HandledBlocks.count(BB))
        continue;

      HandledBlocks.insert(BB);

      // If this BB is too big stop searching to save compiling time.
      if (BB->sizeWithoutDebugLargerThan(MaxBasicBlockSize)) {
        HasConflictCache[BlockPair] = true;
        return true;
      }

      for (MachineInstr &I : *BB) {
        if (I.isCall() || I.hasOrderedMemoryRef()) {
          HasConflictCache[BlockPair] = true;
          return true;
        }

        if (I.mayStore() || I.hasUnmodeledSideEffects()) {
          SawPotentialConflict = true;
          // We still have chance to sink MI if all stores between are not
          // aliased to MI, and neither have side effects.
          // Cache all conflicts, so that we don't need to go through
          // all From reachable blocks for next load instruction.
          if (sideEffectConflict(MI, I) || I.mayAlias(AA, MI, false)) {
            LLVM_DEBUG(dbgs() << " *** Conflict with "; I.print(dbgs()));
            HasConflict = true;
          }
          ConflictInstrCache[BlockPair].push_back(&I);
        }
      }
    }
  }
  // If there is no conflict at all, cache the result.
  if (!SawPotentialConflict)
    HasConflictCache[BlockPair] = false;
  return HasConflict;
}

bool AMDGPUBundleIdxLdSt::sinkInstruction(MachineInstr &MI, bool &SawStore) {

  // Don't sink instructions that the target prefers not to sink.
  if (!TII->shouldSink(MI))
    return false;

  // Check if it's safe to move the instruction.
  if (!MI.isSafeToMove(SawStore))
    return false;

  // Convergent operations may not be made control-dependent on additional
  // values.
  if (MI.isConvergent())
    return false;

  MachineBasicBlock *ParentBlock = MI.getParent();
  SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>, 4>
      SuccsToSinkTo = findSuccsToSinkTo(MI, ParentBlock);

  size_t SinksRemaining = SuccsToSinkTo.size();
  if (SinksRemaining == 0)
    return false;

  LLVM_DEBUG(dbgs() << " *** Found " << SinksRemaining << " use(s)\n");
  for (auto Pair : SuccsToSinkTo) {
    auto Succ = Pair.first;
    auto InsertPos = Pair.second;
    // Note that if we previously encountered Succ == MI.getParent(), we'll
    // have an extra sink remaining, which is need for the remaining local use.
    if (Succ == MI.getParent()) {
      LLVM_DEBUG(
          dbgs()
          << " *** Use is in MI's current block. Leaving a copy in block "
          << Succ->getNumber() << "\n");
      continue;
    }

    if (SinksRemaining > 1) {
      assert(MI.getOpcode() == AMDGPU::V_LOAD_IDX);
      LLVM_DEBUG(dbgs() << "\t *** Duplicating MI and sinking to block "
                        << Succ->getNumber() << "\n");
      MachineInstr *DupLoad =
          MI.getParent()->getParent()->CloneMachineInstr(&MI);
      MI.getParent()->insert(MI, DupLoad);

      // When we duplicate, we must assign to a new register because the
      // bundling phase requires searching for an inst's def, of which there can
      // only be one.
      Register OldDefReg = DupLoad->getOperand(0).getReg();
      auto *RC = MRI->getRegClass(OldDefReg);
      Register NewDefReg = MRI->createVirtualRegister(RC);
      for (auto &UseInSucc : MRI->use_nodbg_operands(OldDefReg)) {
        if (UseInSucc.getParent()->getParent() != Succ || !UseInSucc.isReg() ||
            UseInSucc.getReg() != OldDefReg)
          continue;
        UseInSucc.setReg(NewDefReg);
      }
      DupLoad->getOperand(0).setReg(NewDefReg);
      performSink(*DupLoad, *Succ, InsertPos);
    } else {
      LLVM_DEBUG(dbgs() << "\t *** Sinking MI to block " << Succ->getNumber()
                        << "\n");
      performSink(MI, *Succ, InsertPos);
    }
    SinksRemaining--;
  }

  return true;
}

bool AMDGPUBundleIdxLdSt::sinkLoadsAndCoreMIs(MachineFunction &MF) {
  bool MadeChange = false;
  bool IsConflict = false;
  for (auto &MBB : ReversePostOrderTraversal<MachineFunction *>(&MF)) {

    // Walk the basic block bottom-up.
    SmallVector<MachineInstr *, 8> Conflicts;
    for (auto &I : make_early_inc_range(llvm::reverse(*MBB))) {
      MachineInstr &MI = I; // MI is the instruction to sink.

      // Check if MI conflicts with any of the previously seen instructions in
      // this block
      IsConflict = false;
      for (auto C : Conflicts)
        if (MI.mayAlias(AA, *C, false) || sideEffectConflict(MI, I))
          IsConflict = true;

      if (MI.mayStore() || sideEffectConflict(MI, I))
        Conflicts.push_back(&MI);

      LLVM_DEBUG(dbgs() << "BB." << MBB->getNumber() << " :: ";
                 MI.print(dbgs()));

      if (sinkInstruction(MI, IsConflict))
        MadeChange = true;
    }
  }

  // Now clear any kill flags for recorded registers.
  LLVM_DEBUG(dbgs() << "\n");
  for (auto I : RegsToClearKillFlags)
    MRI->clearKillFlags(I);
  RegsToClearKillFlags.clear();

  return MadeChange;
}

bool AMDGPUBundleIdxLdSt::bundleIdxLdSt(MachineInstr *MI) {
  LLVM_DEBUG(dbgs() << "BB." << MI->getParent()->getNumber() << " :: ";
             MI->print(dbgs()));

  if (MI->isMetaInstruction())
    return false;
  // Prevent cycles in data-flow from multiple defs. This check is too coarse.
  // TODO-GFX13 Handle MI with multiple defs.
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
  // TODO-GFX13 Handle phis.
  if (MI->isPHI())
    return false;

  MachineFunction *MF = MI->getParent()->getParent();
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
    // If we tried to sink it but couldn't, skip.
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
  if (Worklist.size() > 0) {
    bool MILoads = MI->mayLoad();
    assert(!MI->mayStore() || MILoads &&
                                  "Unexpected MI which produces a values and "
                                  "stores but does not load");
    if (MILoads) {
      MachineBasicBlock::iterator I = MI->getIterator(),
                                  E = Worklist[0].MI->getIterator();
      for (++I; I != E; ++I) {
        if (I->mayStore() && MI->mayAlias(AA, *I, false)) {
          LLVM_DEBUG(dbgs() << " *** Conflict with "; I->print(dbgs()));
          return false;
        }
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
          recoverIdx0ForPrivateUse(Worklist, IdxList, StagingRegIdx);
        UsesIdx0ForPrivate = true;
        UsesIdx0ForDynamic = false;
      }
      continue;
    }

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

    // Do not move any V_LOAD_IDX past an aliased V_STORE_IDX.
    bool AliasConflict = false;
    MachineBasicBlock::instr_iterator I = LoadMI->getIterator(),
                                      E = Worklist[0].MI->getIterator();
    for (++I; I != E; ++I) {
      if (I->isBundle())
        I++;
      if (I->mayStore() && LoadMI->mayAlias(AA, *I, false)) {
        LLVM_DEBUG(dbgs() << " *** Conflict with "; I->print(dbgs()));
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
        recoverIdx0ForPrivateUse(Worklist, IdxList, StagingRegIdx);
        UsesIdx0ForDynamic = false;
        UsesIdx0ForPrivate = true;
        continue;
      }
      IdxList.insert(IdxOp->getReg());
    }

    // Duplicate V_LOAD_IDX with uses in multiple instructions.
    auto It = MRI->use_instr_nodbg_begin(UseReg);
    if (++It != MRI->use_instr_nodbg_end()) {
      LLVM_DEBUG(dbgs() << " *** Duplicating "; LoadMI->print(dbgs()));
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
  LLVM_DEBUG({
    dbgs() << " *** Created bundle from \n";
    for (auto Item : reverse(Worklist))
      dbgs() << "\t" << *(Item.MI);
  });
  return true;
}

bool AMDGPUBundleIdxLdSt::runOnMachineFunction(MachineFunction &MF) {

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  if (!ST.hasVGPRIndexingRegisters())
    return false;

  TRI = ST.getRegisterInfo();
  STI = ST.getInstrInfo();
  TII = MF.getSubtarget().getInstrInfo();
  MRI = &MF.getRegInfo();
  if (auto *AAR = getAnalysisIfAvailable<AAResultsWrapperPass>())
    AA = &AAR->getAAResults();
  CI = &getAnalysis<MachineCycleInfoWrapperPass>().getCycleInfo();
  NeedsAlignedVGPRs = ST.needsAlignedVGPRs();

  LLVM_DEBUG(dbgs() << "===== AMDGPUBundleIdxLdSt :: Sinking Phase =====\n");
  bool Changed = sinkLoadsAndCoreMIs(MF);

  LLVM_DEBUG(dbgs() << "===== AMDGPUBundleIdxLdSt :: Bundling Phase =====\n");
  for (MachineBasicBlock &MBB : MF) {
    auto Iter = make_early_inc_range(MBB);
    for (auto &MI : Iter)
      Changed |= bundleIdxLdSt(&MI);
  }
  return Changed;
}

char AMDGPUBundleIdxLdSt::ID = 0;
char &llvm::AMDGPUBundleIdxLdStID = AMDGPUBundleIdxLdSt::ID;

INITIALIZE_PASS_BEGIN(AMDGPUBundleIdxLdSt, DEBUG_TYPE,
                      "Bundle indexed load/store with uses", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineCycleInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUBundleIdxLdSt, DEBUG_TYPE,
                    "Bundle indexed load/store with uses", false, false)
#endif /* LLPC_BUILD_NPI */
