//===-- lib/CodeGen/GlobalISel/Combiner.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file constains common code to combine machine functions at generic
// level.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/CSEMIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/GISelWorkList.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gi-combiner"

using namespace llvm;

STATISTIC(NumOneIteration, "Number of functions with one iteration");
STATISTIC(NumTwoIterations, "Number of functions with two iterations");
STATISTIC(NumThreeOrMoreIterations,
          "Number of functions with three or more iterations");

namespace llvm {
cl::OptionCategory GICombinerOptionCategory(
    "GlobalISel Combiner",
    "Control the rules which are enabled. These options all take a comma "
    "separated list of rules to disable and may be specified by number "
    "or number range (e.g. 1-10)."
#ifndef NDEBUG
    " They may also be specified by name."
#endif
);
} // end namespace llvm

/// This class acts as the glue that joins the CombinerHelper to the overall
/// Combine algorithm. The CombinerHelper is intended to report the
/// modifications it makes to the MIR to the GISelChangeObserver and the
/// observer subclass will act on these events.
class Combiner::WorkListMaintainer : public GISelChangeObserver {
protected:
#ifndef NDEBUG
  /// The instructions that have been created but we want to report once they
  /// have their operands. This is only maintained if debug output is requested.
  SmallSetVector<const MachineInstr *, 32> CreatedInstrs;
#endif
  using Level = CombinerInfo::ObserverLevel;

public:
  static std::unique_ptr<WorkListMaintainer>
  create(Level Lvl, WorkListTy &WorkList, MachineRegisterInfo &MRI);

  virtual ~WorkListMaintainer() = default;

  void reportFullyCreatedInstrs() {
    LLVM_DEBUG({
      for (auto *MI : CreatedInstrs) {
        dbgs() << "Created: " << *MI;
      }
      CreatedInstrs.clear();
    });
  }

  virtual void reset() = 0;
  virtual void appliedCombine() = 0;
};

/// A configurable WorkListMaintainer implementation.
/// The ObserverLevel determines how the WorkListMaintainer reacts to MIR
/// changes.
template <CombinerInfo::ObserverLevel Lvl>
class Combiner::WorkListMaintainerImpl : public Combiner::WorkListMaintainer {
  WorkListTy &WorkList;
  MachineRegisterInfo &MRI;

  // Defer handling these instructions until the combine finishes.
  SmallSetVector<MachineInstr *, 32> DeferList;

  // Track VRegs that (might) have lost a use.
  SmallSetVector<Register, 32> LostUses;

public:
  WorkListMaintainerImpl(WorkListTy &WorkList, MachineRegisterInfo &MRI)
      : WorkList(WorkList), MRI(MRI) {}

  virtual ~WorkListMaintainerImpl() = default;

  void reset() override {
    DeferList.clear();
    LostUses.clear();
  }

  void erasingInstr(MachineInstr &MI) override {
    // MI will become dangling, remove it from all lists.
    LLVM_DEBUG(dbgs() << "Erasing: " << MI; CreatedInstrs.remove(&MI));
    WorkList.remove(&MI);
    if constexpr (Lvl != Level::Basic) {
      DeferList.remove(&MI);
      noteLostUses(MI);
    }
  }

  void createdInstr(MachineInstr &MI) override {
    LLVM_DEBUG(dbgs() << "Creating: " << MI; CreatedInstrs.insert(&MI));
    if constexpr (Lvl == Level::Basic)
      WorkList.insert(&MI);
    else
      // Defer handling newly created instructions, because they don't have
      // operands yet. We also insert them into the WorkList in reverse
      // order so that they will be combined top down.
      DeferList.insert(&MI);
  }

  void changingInstr(MachineInstr &MI) override {
    LLVM_DEBUG(dbgs() << "Changing: " << MI);
    // Some uses might get dropped when MI is changed.
    // For now, overapproximate by assuming all uses will be dropped.
    // TODO: Is a more precise heuristic or manual tracking of use count
    // decrements worth it?
    if constexpr (Lvl != Level::Basic)
      noteLostUses(MI);
  }

  void changedInstr(MachineInstr &MI) override {
    LLVM_DEBUG(dbgs() << "Changed: " << MI);
    if constexpr (Lvl == Level::Basic)
      WorkList.insert(&MI);
    else
      // Defer this for DCE
      DeferList.insert(&MI);
  }

  // Only track changes during the combine and then walk the def/use-chains once
  // the combine is finished, because:
  // - instructions might have multiple defs during the combine.
  // - use counts aren't accurate during the combine.
  void appliedCombine() override {
    if constexpr (Lvl == Level::Basic)
      return;

    // DCE deferred instructions and add them to the WorkList bottom up.
    while (!DeferList.empty()) {
      MachineInstr &MI = *DeferList.pop_back_val();
      if (tryDCE(MI, MRI))
        continue;

      if constexpr (Lvl >= Level::SinglePass)
        addUsersToWorkList(MI);

      WorkList.insert(&MI);
    }

    // Handle instructions that have lost a user.
    while (!LostUses.empty()) {
      Register Use = LostUses.pop_back_val();
      MachineInstr *UseMI = MRI.getVRegDef(Use);
      if (!UseMI)
        continue;

      // If DCE succeeds, UseMI's uses are added back to LostUses by
      // erasingInstr.
      if (tryDCE(*UseMI, MRI))
        continue;

      if constexpr (Lvl >= Level::SinglePass) {
        // OneUse checks are relatively common, so we might be able to combine
        // the single remaining user of this Reg.
        if (MRI.hasOneNonDBGUser(Use))
          WorkList.insert(&*MRI.use_instr_nodbg_begin(Use));

        WorkList.insert(UseMI);
      }
    }
  }

  void noteLostUses(MachineInstr &MI) {
    for (auto &Use : MI.explicit_uses()) {
      if (!Use.isReg() || !Use.getReg().isVirtual())
        continue;
      LostUses.insert(Use.getReg());
    }
  }

  void addUsersToWorkList(MachineInstr &MI) {
    for (auto &Def : MI.defs()) {
      Register DefReg = Def.getReg();
      if (!DefReg.isVirtual())
        continue;
      for (auto &UseMI : MRI.use_nodbg_instructions(DefReg)) {
        WorkList.insert(&UseMI);
      }
    }
  }
};

std::unique_ptr<Combiner::WorkListMaintainer>
Combiner::WorkListMaintainer::create(Level Lvl, WorkListTy &WorkList,
                                     MachineRegisterInfo &MRI) {
  switch (Lvl) {
  case Level::Basic:
    return std::make_unique<WorkListMaintainerImpl<Level::Basic>>(WorkList,
                                                                  MRI);
  case Level::DCE:
    return std::make_unique<WorkListMaintainerImpl<Level::DCE>>(WorkList, MRI);
  case Level::SinglePass:
    return std::make_unique<WorkListMaintainerImpl<Level::SinglePass>>(WorkList,
                                                                       MRI);
  }
  llvm_unreachable("Illegal ObserverLevel");
}

Combiner::Combiner(MachineFunction &MF, CombinerInfo &CInfo,
                   const TargetPassConfig *TPC, GISelValueTracking *VT,
                   GISelCSEInfo *CSEInfo)
    : Builder(CSEInfo ? std::make_unique<CSEMIRBuilder>()
                      : std::make_unique<MachineIRBuilder>()),
      WLObserver(WorkListMaintainer::create(CInfo.ObserverLvl, WorkList,
                                            MF.getRegInfo())),
      ObserverWrapper(std::make_unique<GISelObserverWrapper>()), CInfo(CInfo),
      Observer(*ObserverWrapper), B(*Builder), MF(MF), MRI(MF.getRegInfo()),
      VT(VT), TPC(TPC), CSEInfo(CSEInfo) {
  (void)this->TPC; // FIXME: Remove when used.

  // Setup builder.
  B.setMF(MF);
  if (CSEInfo)
    B.setCSEInfo(CSEInfo);

  B.setChangeObserver(*ObserverWrapper);
}

Combiner::~Combiner() = default;

bool Combiner::tryDCE(MachineInstr &MI, MachineRegisterInfo &MRI) {
  if (!isTriviallyDead(MI, MRI))
    return false;
  LLVM_DEBUG(dbgs() << "Dead: " << MI);
  llvm::salvageDebugInfo(MRI, MI);
  MI.eraseFromParent();
  return true;
}

bool Combiner::combineMachineInstrs() {
  // If the ISel pipeline failed, do not bother running this pass.
  // FIXME: Should this be here or in individual combiner passes.
  if (MF.getProperties().hasFailedISel())
    return false;

  // We can't call this in the constructor because the derived class is
  // uninitialized at that time.
  if (!HasSetupMF) {
    HasSetupMF = true;
    setupMF(MF, VT);
  }

  LLVM_DEBUG(dbgs() << "Generic MI Combiner for: " << MF.getName() << '\n');

  MachineOptimizationRemarkEmitter MORE(MF, /*MBFI=*/nullptr);

  bool MFChanged = false;
  bool Changed;

  unsigned Iteration = 0;
  while (true) {
    ++Iteration;
    LLVM_DEBUG(dbgs() << "\n\nCombiner iteration #" << Iteration << '\n');

    Changed = false;
    WorkList.clear();
    WLObserver->reset();
    ObserverWrapper->clearObservers();
    if (CSEInfo)
      ObserverWrapper->addObserver(CSEInfo);

    // If Observer-based DCE is enabled, perform full DCE only before the first
    // iteration.
    bool EnableDCE = CInfo.ObserverLvl >= CombinerInfo::ObserverLevel::DCE
                         ? CInfo.EnableFullDCE && Iteration == 1
                         : CInfo.EnableFullDCE;

    // Collect all instructions. Do a post order traversal for basic blocks and
    // insert with list bottom up, so while we pop_back_val, we'll traverse top
    // down RPOT.
    RAIIMFObsDelInstaller DelInstall(MF, *ObserverWrapper);
    for (MachineBasicBlock *MBB : post_order(&MF)) {
      for (MachineInstr &CurMI :
           llvm::make_early_inc_range(llvm::reverse(*MBB))) {
        // Erase dead insts before even adding to the list.
        if (EnableDCE && tryDCE(CurMI, MRI))
          continue;
        WorkList.deferred_insert(&CurMI);
      }
    }
    WorkList.finalize();

    // Only notify WLObserver during actual combines
    ObserverWrapper->addObserver(WLObserver.get());
    // Main Loop. Process the instructions here.
    while (!WorkList.empty()) {
      MachineInstr &CurrInst = *WorkList.pop_back_val();
      LLVM_DEBUG(dbgs() << "\nTry combining " << CurrInst);
      bool AppliedCombine = tryCombineAll(CurrInst);
      LLVM_DEBUG(WLObserver->reportFullyCreatedInstrs());
      Changed |= AppliedCombine;
      if (AppliedCombine)
        WLObserver->appliedCombine();
    }
    MFChanged |= Changed;

    if (!Changed) {
      LLVM_DEBUG(dbgs() << "\nCombiner reached fixed-point after iteration #"
                        << Iteration << '\n');
      break;
    }
    // Iterate until a fixed-point is reached if MaxIterations == 0,
    // otherwise limit the number of iterations.
    if (CInfo.MaxIterations && Iteration >= CInfo.MaxIterations) {
      LLVM_DEBUG(
          dbgs() << "\nCombiner reached iteration limit after iteration #"
                 << Iteration << '\n');
      break;
    }
  }

  if (Iteration == 1)
    ++NumOneIteration;
  else if (Iteration == 2)
    ++NumTwoIterations;
  else
    ++NumThreeOrMoreIterations;

#ifndef NDEBUG
  if (CSEInfo) {
    if (auto E = CSEInfo->verify()) {
      errs() << E << '\n';
      assert(false && "CSEInfo is not consistent. Likely missing calls to "
                      "observer on mutations.");
    }
  }
#endif
  return MFChanged;
}
