//===-- SIPostRABundler.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass creates bundles of memory instructions to protect adjacent loads
/// and stores from being rescheduled apart from each other post-RA.
///
//===----------------------------------------------------------------------===//

#include "SIPostRABundler.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <deque>

using namespace llvm;

#define DEBUG_TYPE "si-post-ra-bundler"

namespace {

class SIPostRABundlerLegacy : public MachineFunctionPass {
public:
  static char ID;

public:
  SIPostRABundlerLegacy() : MachineFunctionPass(ID) {
    initializeSIPostRABundlerLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI post-RA bundler";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class SIPostRABundler {
public:
  bool run(MachineFunction &MF);

private:
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI;

  SmallSet<Register, 16> Defs;

  void collectUsedRegUnits(const MachineInstr &MI,
                           BitVector &UsedRegUnits) const;

  bool isBundleCandidate(const MachineInstr &MI) const;
  bool isDependentLoad(const MachineInstr &MI) const;
  bool canBundle(const MachineInstr &MI, const MachineInstr &NextMI) const;
  void reorderLoads(MachineBasicBlock &MBB,
                    MachineBasicBlock::instr_iterator &BundleStart,
                    MachineBasicBlock::instr_iterator Next);
};

constexpr uint64_t MemFlags = SIInstrFlags::MTBUF | SIInstrFlags::MUBUF |
                              SIInstrFlags::SMRD | SIInstrFlags::DS |
                              SIInstrFlags::FLAT | SIInstrFlags::MIMG |
                              SIInstrFlags::VIMAGE | SIInstrFlags::VSAMPLE;

} // End anonymous namespace.

INITIALIZE_PASS(SIPostRABundlerLegacy, DEBUG_TYPE, "SI post-RA bundler", false,
                false)

char SIPostRABundlerLegacy::ID = 0;

char &llvm::SIPostRABundlerLegacyID = SIPostRABundlerLegacy::ID;

FunctionPass *llvm::createSIPostRABundlerPass() {
  return new SIPostRABundlerLegacy();
}

bool SIPostRABundler::isDependentLoad(const MachineInstr &MI) const {
  if (!MI.mayLoad())
    return false;

  for (const MachineOperand &Op : MI.explicit_operands()) {
    if (!Op.isReg())
      continue;
    Register Reg = Op.getReg();
    for (Register Def : Defs)
      if (TRI->regsOverlap(Reg, Def))
        return true;
  }

  return false;
}

void SIPostRABundler::collectUsedRegUnits(const MachineInstr &MI,
                                          BitVector &UsedRegUnits) const {
  if (MI.isDebugInstr())
    return;

  for (const MachineOperand &Op : MI.operands()) {
    if (!Op.isReg() || !Op.readsReg())
      continue;

    Register Reg = Op.getReg();
    assert(!Op.getSubReg() &&
           "subregister indexes should not be present after RA");

    for (MCRegUnit Unit : TRI->regunits(Reg))
      UsedRegUnits.set(static_cast<unsigned>(Unit));
  }
}

bool SIPostRABundler::isBundleCandidate(const MachineInstr &MI) const {
  const uint64_t IMemFlags = MI.getDesc().TSFlags & MemFlags;
  return IMemFlags != 0 && MI.mayLoadOrStore() && !MI.isBundled();
}

bool SIPostRABundler::canBundle(const MachineInstr &MI,
                                const MachineInstr &NextMI) const {
  const uint64_t IMemFlags = MI.getDesc().TSFlags & MemFlags;

  return (IMemFlags != 0 && MI.mayLoadOrStore() && !NextMI.isBundled() &&
          NextMI.mayLoad() == MI.mayLoad() && NextMI.mayStore() == MI.mayStore() &&
          ((NextMI.getDesc().TSFlags & MemFlags) == IMemFlags) &&
          !isDependentLoad(NextMI));
}

static Register getDef(MachineInstr &MI) {
  assert(MI.getNumExplicitDefs() > 0);
  return MI.defs().begin()->getReg();
}

void SIPostRABundler::reorderLoads(
    MachineBasicBlock &MBB, MachineBasicBlock::instr_iterator &BundleStart,
    MachineBasicBlock::instr_iterator Next) {
  // Don't reorder ALU, store or scalar clauses.
  if (!BundleStart->mayLoad() || BundleStart->mayStore() ||
      SIInstrInfo::isSMRD(*BundleStart) || !BundleStart->getNumExplicitDefs())
    return;

  // Search to find the usage distance of each defined register in the clause.
  const unsigned SearchDistance = std::max(Defs.size(), 100UL);
  SmallDenseMap<Register, unsigned> UseDistance;
  unsigned MaxDistance = 0;
  for (MachineBasicBlock::iterator SearchI = Next;
       SearchI != MBB.end() && MaxDistance < SearchDistance &&
       UseDistance.size() < Defs.size();
       ++SearchI, ++MaxDistance) {
    for (Register Reg : Defs) {
      if (UseDistance.contains(Reg))
        continue;
      if (SearchI->readsRegister(Reg, TRI))
        UseDistance[Reg] = MaxDistance;
    }
  }

  if (UseDistance.empty())
    return;

  LLVM_DEBUG(dbgs() << "Try bundle reordering\n");

  // Build schedule based on use distance of register uses.
  // Attempt to preserve exist order (NativeOrder) where possible.
  std::deque<std::pair<MachineInstr *, unsigned>> Schedule;
  unsigned NativeOrder = 0, LastOrder = 0;
  bool Reordered = false;
  for (auto II = BundleStart; II != Next; ++II, ++NativeOrder) {
    // Bail out if we encounter anything that seems risky to reorder.
    if (!II->getNumExplicitDefs() || II->isKill() ||
        llvm::any_of(II->memoperands(), [&](const MachineMemOperand *MMO) {
          return MMO->isAtomic() || MMO->isVolatile();
        })) {
      LLVM_DEBUG(dbgs() << " Abort\n");
      return;
    }

    Register Reg = getDef(*II);
    unsigned NewOrder =
        UseDistance.contains(Reg) ? UseDistance[Reg] : MaxDistance;
    LLVM_DEBUG(dbgs() << "  Order: " << NewOrder << "," << NativeOrder
                      << ", MI: " << *II);
    unsigned Order = (NewOrder << 16 | NativeOrder);
    Schedule.emplace_back(&*II, Order);
    Reordered |= Order < LastOrder;
    LastOrder = Order;
  }

  // No reordering found.
  if (!Reordered) {
    LLVM_DEBUG(dbgs() << " No changes\n");
    return;
  }

  // Apply sort on new ordering.
  std::sort(Schedule.begin(), Schedule.end(),
            [](std::pair<MachineInstr *, unsigned> A,
               std::pair<MachineInstr *, unsigned> B) {
              return A.second < B.second;
            });

  // Rebuild clause order.
  // Schedule holds ideal order for the load operations; however, each def
  // can only be scheduled when it will no longer clobber any uses.
  SmallVector<MachineInstr *> Clause;
  while (!Schedule.empty()) {
    // Try to schedule next instruction in schedule.
    // Iterate until we find something that can be placed.
    auto It = Schedule.begin();
    while (It != Schedule.end()) {
      MachineInstr *MI = It->first;
      LLVM_DEBUG(dbgs() << "Try schedule: " << *MI);

      if (MI->getNumExplicitDefs() == 0) {
        // No defs, always schedule.
        LLVM_DEBUG(dbgs() << "  Trivially OK\n");
        break;
      }

      Register DefReg = getDef(*MI);
      bool DefRegHasUse = false;
      for (auto SearchIt = std::next(It);
           SearchIt != Schedule.end() && !DefRegHasUse; ++SearchIt)
        DefRegHasUse = SearchIt->first->readsRegister(DefReg, TRI);
      if (DefRegHasUse) {
        // A future use would be clobbered; try next instruction in the
        // schedule.
        LLVM_DEBUG(dbgs() << "  Clobbers uses\n");
        It++;
        continue;
      }

      // Safe to schedule.
      LLVM_DEBUG(dbgs() << "  OK!\n");
      break;
    }

    // Place schedule instruction into clause order.
    assert(It != Schedule.end());
    MachineInstr *MI = It->first;
    Schedule.erase(It);
    Clause.push_back(MI);

    // Clear kill flags for later uses.
    for (auto &Use : MI->all_uses()) {
      if (!Use.isReg() || !Use.isKill())
        continue;
      Register UseReg = Use.getReg();
      if (llvm::any_of(Schedule, [&](std::pair<MachineInstr *, unsigned> &SI) {
            return SI.first->readsRegister(UseReg, TRI);
          }))
        Use.setIsKill(false);
    }
  }

  // Apply order to instructions.
  for (MachineInstr *MI : Clause)
    MI->moveBefore(&*Next);

  // Update start of bundle.
  BundleStart = Clause[0]->getIterator();
}

bool SIPostRABundlerLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  return SIPostRABundler().run(MF);
}

PreservedAnalyses SIPostRABundlerPass::run(MachineFunction &MF,
                                           MachineFunctionAnalysisManager &) {
  SIPostRABundler().run(MF);
  return PreservedAnalyses::all();
}

bool SIPostRABundler::run(MachineFunction &MF) {

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  BitVector BundleUsedRegUnits(TRI->getNumRegUnits());
  BitVector KillUsedRegUnits(TRI->getNumRegUnits());

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    bool HasIGLPInstrs = llvm::any_of(MBB.instrs(), [](MachineInstr &MI) {
      unsigned Opc = MI.getOpcode();
      return Opc == AMDGPU::SCHED_GROUP_BARRIER || Opc == AMDGPU::IGLP_OPT;
    });

    // Don't cluster with IGLP instructions.
    if (HasIGLPInstrs)
      continue;

    MachineBasicBlock::instr_iterator Next;
    MachineBasicBlock::instr_iterator B = MBB.instr_begin();
    MachineBasicBlock::instr_iterator E = MBB.instr_end();

    for (auto I = B; I != E; I = Next) {
      Next = std::next(I);
      if (!isBundleCandidate(*I))
        continue;

      assert(Defs.empty());

      if (I->getNumExplicitDefs() != 0)
        Defs.insert(getDef(*I));

      MachineBasicBlock::instr_iterator BundleStart = I;
      MachineBasicBlock::instr_iterator BundleEnd = I;
      unsigned ClauseLength = 1;
      for (I = Next; I != E; I = Next) {
        Next = std::next(I);

        assert(BundleEnd != I);
        if (canBundle(*BundleEnd, *I)) {
          BundleEnd = I;
          if (I->getNumExplicitDefs() != 0)
            Defs.insert(getDef(*I));
          ++ClauseLength;
        } else if (!I->isMetaInstruction() ||
                   I->getOpcode() == AMDGPU::SCHED_BARRIER) {
          // SCHED_BARRIER is not bundled to be honored by scheduler later.
          // Allow other meta instructions in between bundle candidates, but do
          // not start or end a bundle on one.
          //
          // TODO: It may be better to move meta instructions like dbg_value
          // after the bundle. We're relying on the memory legalizer to unbundle
          // these.
          break;
        }
      }

      Next = std::next(BundleEnd);
      if (ClauseLength > 1) {
        Changed = true;

        // Before register allocation, kills are inserted after potential soft
        // clauses to hint register allocation. Look for kills that look like
        // this, and erase them.
        if (Next != E && Next->isKill()) {

          // TODO: Should maybe back-propagate kill flags to the bundle.
          for (const MachineInstr &BundleMI : make_range(BundleStart, Next))
            collectUsedRegUnits(BundleMI, BundleUsedRegUnits);

          BundleUsedRegUnits.flip();

          while (Next != E && Next->isKill()) {
            MachineInstr &Kill = *Next;
            collectUsedRegUnits(Kill, KillUsedRegUnits);

            KillUsedRegUnits &= BundleUsedRegUnits;

            // Erase the kill if it's a subset of the used registers.
            //
            // TODO: Should we just remove all kills? Is there any real reason to
            // keep them after RA?
            if (KillUsedRegUnits.none()) {
              ++Next;
              Kill.eraseFromParent();
            } else
              break;

            KillUsedRegUnits.reset();
          }

          BundleUsedRegUnits.reset();
        }

        reorderLoads(MBB, BundleStart, Next);
        finalizeBundle(MBB, BundleStart, Next);
      }

      Defs.clear();
    }
  }

  return Changed;
}
