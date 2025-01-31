//===- Spill2Reg.cpp - Spill To Register Optimization ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file implements Spill2Reg, an optimization which selectively
/// replaces spills/reloads to/from the stack with register copies to/from other
/// registers. This works even on targets where load/stores have similar latency
/// to register copies because it can free up memory units which helps avoid
/// stalls in the pipeline.
///
//===----------------------------------------------------------------------===//

#include "AllocationOrder.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "Spill2Reg"
STATISTIC(NumSpill2RegInstrs, "Number of spills/reloads replaced by spill2reg");

namespace {

class Spill2Reg : public MachineFunctionPass {
public:
  static char ID;
  Spill2Reg() : MachineFunctionPass(ID) {
    initializeSpill2RegPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  bool runOnMachineFunction(MachineFunction &) override;

private:
  /// Holds data for spills and reloads.
  struct StackSlotDataEntry {
    /// This is set to true to disable code generation for the spills/reloads
    /// that we collected in this entry.
    bool Disable = false;
    /// Indentation for the dump() methods.
    static constexpr const int DumpInd = 2;

    /// The data held for each spill/reload.
    struct MIData {
      MIData(MachineInstr *MI, const MachineOperand *MO, unsigned SpillBits)
          : MI(MI), MO(MO), SpillBits(SpillBits) {}
      /// The Spill/Reload instruction.
      MachineInstr *MI = nullptr;
      /// The operand being spilled/reloaded.
      const MachineOperand *MO = nullptr;
      /// The size of the data spilled/reloaded in bits. This occasionally
      /// differs across accesses to the same stack slot.
      unsigned SpillBits = 0;
#ifndef NDEBUG
      LLVM_DUMP_METHOD virtual void dump() const;
      virtual ~MIData() {}
#endif
    };

    struct MIDataWithLiveIn : public MIData {
      MIDataWithLiveIn(MachineInstr *MI, const MachineOperand *MO,
                       unsigned SpillBits)
          : MIData(MI, MO, SpillBits) {}
      /// We set this to false to mark the vector register associated to this
      /// reload as definitely not live-in. This is useful in blocks with both
      /// spill and reload of the same stack slot, like in the example:
      /// \verbatim
      ///  bb:
      ///    spill %stack.0
      ///    reload %stack.0
      /// \endverbatim
      /// This information is used during `updateLiveIns()`. We are collecting
      /// this information during `collectSpillsAndReloads()` because we are
      /// already walking through the code there. Otherwise we would need to
      /// walk throught the code again in `updateLiveIns()` just to check for
      /// other spills in the block, which would waste compilation time.
      bool IsLiveIn = true;
#ifndef NDEBUG
      LLVM_DUMP_METHOD virtual void dump() const override;
#endif
    };
    SmallVector<MIData, 1> Spills;
    SmallVector<MIDataWithLiveIn, 1> Reloads;

    /// \Returns the register class of the register being spilled.
    const TargetRegisterClass *
    getSpilledRegClass(const TargetInstrInfo *TII,
                       const TargetRegisterInfo *TRI,
                       const TargetSubtargetInfo *STI) const {
      auto Reg0 = Spills.front().MO->getReg();
      return TRI->getCandidateRegisterClassForSpill2Reg(TRI, STI, Reg0);
    }

#ifndef NDEBUG
    LLVM_DUMP_METHOD void dump() const;
#endif
  };
  /// Look for candidates for spill2reg. These candidates are in places with
  /// high memory unit contention. Fills in StackSlotData.
  void collectSpillsAndReloads();
  /// \Returns if \p MI is profitable to apply spill-to-reg by checking whether
  /// this would remove pipeline bubbles.
  bool isProfitable(const MachineInstr *MI) const;
  /// \Returns true if any stack-based spill/reload in \p Entry is profitable
  /// to replace with a reg-based spill/reload.
  bool allAccessesProfitable(const StackSlotDataEntry &Entry) const;
  /// Look for a free physical register in \p LRU of reg class \p RegClass.
  std::optional<MCRegister>
  tryGetFreePhysicalReg(const TargetRegisterClass *RegClass,
                        const LiveRegUnits &LRU);
  /// Helper for generateCode(). It eplaces stack spills or reloads with movs
  /// to \p LI.reg().
  void replaceStackWithReg(StackSlotDataEntry &Entry, Register VectorReg);
  /// Updates the live-ins of MBBs after we emit the new spill2reg instructions
  /// and the vector registers become live from register spills to reloads.
  void updateLiveIns(StackSlotDataEntry &Entry, MCRegister VectorReg);
  /// Updates \p LRU with the liveness of physical registers around the spills
  /// and reloads in \p Entry.
  void calculateLiveRegs(StackSlotDataEntry &Entry, LiveRegUnits &LRU);
  /// Replace spills to stack with spills to registers (same for reloads).
  void generateCode();
  /// Cleanup data structures once the pass is finished.
  void cleanup();
  /// The main entry point for this pass.
  bool run();

  /// Map from a stack slot to the corresponding spills and reloads.
  DenseMap<int, StackSlotDataEntry> StackSlotData;
  /// The registers used by each block (from LiveRegUnits). This is needed for
  /// finding free physical registers in the generateCode().
  DenseMap<const MachineBasicBlock *, LiveRegUnits> LRUs;

  MachineFunction *MF = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineFrameInfo *MFI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  RegisterClassInfo RegClassInfo;
};

} // namespace

void Spill2Reg::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void Spill2Reg::releaseMemory() {}

bool Spill2Reg::runOnMachineFunction(MachineFunction &MFn) {
  // Disable if NoImplicitFloat to avoid emitting instrs that use vectors.
  if (MFn.getFunction().hasFnAttribute(Attribute::NoImplicitFloat))
    return false;

  MF = &MFn;
  MRI = &MF->getRegInfo();
  MFI = &MF->getFrameInfo();
  TII = MF->getSubtarget().getInstrInfo();
  TRI = MF->getSubtarget().getRegisterInfo();
  // Enable only if the target supports the appropriate vector instruction set.
  if (!TRI->targetSupportsSpill2Reg(&MF->getSubtarget()))
    return false;

  RegClassInfo.runOnMachineFunction(MFn);

  return run();
}

char Spill2Reg::ID = 0;

char &llvm::Spill2RegID = Spill2Reg::ID;

void Spill2Reg::collectSpillsAndReloads() {
  /// The checks for collecting spills and reloads are identical, so we keep
  /// them here in one place. Return true if we should not collect this.
  auto SkipEntry = [this](int StackSlot, Register Reg) -> bool {
    // If not a spill/reload stack slot.
    if (!MFI->isSpillSlotObjectIndex(StackSlot))
      return true;
    // Check size in bits.
    if (!TRI->isLegalToSpill2Reg(Reg, TRI, MRI))
      return true;
    return false;
  };

  // Collect spills and reloads and associate them to stack slots.
  // If any spill/reload for a stack slot is found not to be eligible for
  // spill-to-reg, then that stack slot is disabled.
  for (MachineBasicBlock &MBB : *MF) {
    // Initialize AccumMBBLRU for keeping track of physical registers used
    // across the whole MBB.
    LiveRegUnits AccumMBBLRU(*TRI);
    AccumMBBLRU.addLiveOuts(MBB);

    // Collect spills/reloads
    for (MachineInstr &MI : llvm::reverse(MBB)) {
      // Update the LRU state as we move upwards.
      AccumMBBLRU.accumulate(MI);

      int StackSlot;
      if (const MachineOperand *MO = TII->isStoreToStackSlotMO(MI, StackSlot)) {
        MachineInstr *Spill = &MI;
        auto &Entry = StackSlotData[StackSlot];
        if (Entry.Disable || SkipEntry(StackSlot, MO->getReg())) {
          Entry.Disable = true;
          continue;
        }
        unsigned SpillBits = TRI->getRegSizeInBits(MO->getReg(), *MRI);
        Entry.Spills.emplace_back(Spill, MO, SpillBits);

        // If any of the reloads collected so far is in the same MBB then mark
        // it as non live-in. This is used in `updateLiveIns()` where we update
        // the liveins of MBBs to include the new vector register. Doing this
        // now avoids an MBB walk in `updateLiveIns()` which should save
        // compilation time.
        // TODO: Perhaps use a MapVector for Entry.Reloads for fast lookup?
        for (auto &MID : Entry.Reloads)
          if (MID.MI->getParent() == &MBB)
            MID.IsLiveIn = false;
      } else if (const MachineOperand *MO =
                     TII->isLoadFromStackSlotMO(MI, StackSlot)) {
        MachineInstr *Reload = &MI;
        auto &Entry = StackSlotData[StackSlot];
        if (Entry.Disable || SkipEntry(StackSlot, MO->getReg())) {
          Entry.Disable = true;
          continue;
        }
        assert(Reload->getRestoreSize(TII) && "Expected reload");
        unsigned SpillBits = TRI->getRegSizeInBits(MO->getReg(), *MRI);
        Entry.Reloads.emplace_back(Reload, MO, SpillBits);

        // Even though the default value of `IsLiveIn` is true, we still need to
        // eagerly mark the reloads in this BB as live-in. This is needed when
        // we have multiple reloads from the same slot in the same BB with
        // spills to the same slot in between that are set to false when
        // visiting a reload.
        //   reload %stack.0 <- IsLiveIn = true
        //   spill  %stack.0 <- IsLiveIn = false
        //   reload %stack.0 <- IsLiveIn = true
        for (auto &MID : Entry.Reloads)
          if (MID.MI->getParent() == &MBB)
            MID.IsLiveIn = true;
      } else {
        // This should capture uses of the stack in instructions that access
        // memory (e.g., folded spills/reloads) and non-memory instructions,
        // like x86 LEA.
        for (const MachineOperand &MO : MI.operands())
          if (MO.isFI()) {
            int StackSlot = MO.getIndex();
            auto &Entry = StackSlotData[StackSlot];
            Entry.Disable = true;
          }
      }
    }

    LRUs.insert(std::make_pair(&MBB, AccumMBBLRU));
  }
}

bool Spill2Reg::isProfitable(const MachineInstr *MI) const {
  return TII->isSpill2RegProfitable(MI, TRI, MRI);
}

bool Spill2Reg::allAccessesProfitable(const StackSlotDataEntry &Entry) const {
  auto IsProfitable = [this](const auto &MID) { return isProfitable(MID.MI); };
  return llvm::all_of(Entry.Spills, IsProfitable) &&
         llvm::all_of(Entry.Reloads, IsProfitable);
}

std::optional<MCRegister>
Spill2Reg::tryGetFreePhysicalReg(const TargetRegisterClass *RegClass,
                                 const LiveRegUnits &LRU) {
  auto Order = RegClassInfo.getOrder(RegClass);
  for (auto I = Order.begin(), E = Order.end(); I != E; ++I) {
    MCRegister PhysVectorReg = *I;
    if (LRU.available(PhysVectorReg))
      return PhysVectorReg;
  }
  return std::nullopt;
}

/// Perform a bottom-up depth-first traversal from \p MBB at \p MI towards its
/// predecessors blocks. Visited marks the visited blocks. \p Fn is the
/// callback function called in pre-order. If \p Fn returns true we stop the
/// traversal.
// TODO: Use df_iterator
static void DFS(MachineBasicBlock *MBB, DenseSet<MachineBasicBlock *> &Visited,
                std::function<bool(MachineBasicBlock *)> Fn) {
  // Skip visited to avoid infinite loops.
  if (Visited.count(MBB))
    return;
  Visited.insert(MBB);

  // Preorder.
  if (Fn(MBB))
    return;

  // Depth-first across predecessors.
  for (MachineBasicBlock *PredMBB : MBB->predecessors())
    DFS(PredMBB, Visited, Fn);
}

void Spill2Reg::updateLiveIns(StackSlotDataEntry &Entry, MCRegister VectorReg) {
  // Collect the parent MBBs of Spills for fast lookup.
  DenseSet<MachineBasicBlock *> SpillMBBs(Entry.Spills.size());
  DenseSet<MachineInstr *> Spills(Entry.Spills.size());
  for (const auto &Data : Entry.Spills) {
    SpillMBBs.insert(Data.MI->getParent());
    Spills.insert(Data.MI);
  }

  auto AddLiveInIfRequired = [VectorReg, &SpillMBBs](MachineBasicBlock *MBB) {
    // If there is a spill in this MBB then we don't need to add a live-in.
    // This works even if there is a reload above the spill, like this:
    //   reload stack.0
    //   spill  stack.0
    // because the live-in due to the reload is handled at a separate walk.
    if (SpillMBBs.count(MBB))
      // Return true to stop the recursion.
      return true;
    // If there are no spills in this block then the register is live-in.
    if (!MBB->isLiveIn(VectorReg))
      MBB->addLiveIn(VectorReg);
    // Return false to continue the recursion.
    return false;
  };

  // Update the MBB live-ins. These are used for the live regs calculation.
  DenseSet<MachineBasicBlock *> Visited;
  for (const auto &ReloadData : Entry.Reloads) {
    MachineInstr *Reload = ReloadData.MI;
    MachineBasicBlock *MBB = Reload->getParent();
    // From a previous walk in MBB we know whether the reload is live-in, or
    // whether the value comes from an earlier spill in the same MBB.
    if (!ReloadData.IsLiveIn)
      continue;
    if (!MBB->isLiveIn(VectorReg))
      MBB->addLiveIn(VectorReg);

    for (MachineBasicBlock *PredMBB : Reload->getParent()->predecessors())
      DFS(PredMBB, Visited, AddLiveInIfRequired);
  }
}

// Replace stack-based spills/reloads with register-based ones.
void Spill2Reg::replaceStackWithReg(StackSlotDataEntry &Entry,
                                    Register VectorReg) {
  for (StackSlotDataEntry::MIData &SpillData : Entry.Spills) {
    MachineInstr *StackSpill = SpillData.MI;
    assert(SpillData.MO->isReg() && "Expected register MO");
    Register OldReg = SpillData.MO->getReg();

    TII->spill2RegInsertToS2RReg(
        VectorReg, OldReg, SpillData.SpillBits, StackSpill->getParent(),
        /*InsertBeforeIt=*/StackSpill->getIterator(), TRI, &MF->getSubtarget());

    // Mark VectorReg as live in the instr's BB.
    LRUs[StackSpill->getParent()].addReg(VectorReg);

    // Spill to stack is no longer needed.
    StackSpill->eraseFromParent();
    assert(OldReg.isPhysical() && "Otherwise we need to removeInterval()");
  }

  for (StackSlotDataEntry::MIData &ReloadData : Entry.Reloads) {
    MachineInstr *StackReload = ReloadData.MI;
    assert(ReloadData.MO->isReg() && "Expected Reg MO");
    Register OldReg = ReloadData.MO->getReg();

    TII->spill2RegExtractFromS2RReg(
        OldReg, VectorReg, ReloadData.SpillBits, StackReload->getParent(),
        /*InsertBeforeIt=*/StackReload->getIterator(), TRI,
        &MF->getSubtarget());

    // Mark VectorReg as live in the instr's BB.
    LRUs[StackReload->getParent()].addReg(VectorReg);

    // Reload from stack is no longer needed.
    StackReload->eraseFromParent();
    assert(OldReg.isPhysical() && "Otherwise we need to removeInterval()");
  }
}

void Spill2Reg::calculateLiveRegs(StackSlotDataEntry &Entry,
                                  LiveRegUnits &LRU) {
  // Collect the parent MBBs of Spills for fast lookup.
  DenseSet<MachineBasicBlock *> SpillMBBs(Entry.Spills.size());
  DenseSet<MachineInstr *> Spills(Entry.Spills.size());
  for (const auto &Data : Entry.Spills) {
    SpillMBBs.insert(Data.MI->getParent());
    Spills.insert(Data.MI);
  }

  /// Walk up the instructions in \p MI's block, accumulating the used registers
  /// into \p LRU, and stopping at a spill or the top of the BB.
  /// \Returns true if a spill was found, false otherwise.
  /// This is used for computing the registers in two cases:
  /// 1. between a reload and a spill (or BB top), and
  /// 2. from the bottom of the BB to a spill.
  //   bb:
  //     ...
  //     spill ^
  //     ...   | Accumulate from MI to spill, or top of the BB.
  //     MI    -
  auto AccumulateLRUUntilSpillFn = [&Spills, &SpillMBBs](MachineInstr *MI,
                                                         LiveRegUnits &LRU) {
    MachineBasicBlock *MBB = MI->getParent();
    bool IsSpillBlock = SpillMBBs.count(MBB);
    // Else walk up the BB, starting from MI, looking for any spill.
    for (MachineInstr *CurrMI = MI; CurrMI != nullptr;
         CurrMI = CurrMI->getPrevNode()) {
      LRU.accumulate(*CurrMI);
      // If a spill is found then return true to end the recursion.
      if (IsSpillBlock && Spills.count(CurrMI))
        return true;
    }
    return false;
  };

  // Accumulates all register units used in \p MBB. If the block contains a
  // spill we walk from the bottom to the spill. If it's an intermediate block,
  // we get the registers from the LRUs map. \Return true once a spill is found.
  auto AccumulateLRUFn = [&SpillMBBs, &LRU, AccumulateLRUUntilSpillFn,
                          this](MachineBasicBlock *MBB) {
    if (SpillMBBs.count(MBB)) {
      // If this is a spill block, then walk bottom-up until the spill.
      assert(!MBB->empty() && "How can it be a spill block and empty?");
      // Add all MBB's live-outs.
      LRU.addLiveOuts(*MBB);
      bool FoundSpill = AccumulateLRUUntilSpillFn(&*MBB->rbegin(), LRU);
      assert(FoundSpill && "Spill block but we couldn't find spill!");
      // We return true to stop the recursion.
      return true;
    }
    // Else this is an intermediate block between the spills and reloads and
    // there is no spill in it, then use the pre-computed LRU to avoid walking
    // it again. This improves compilation time.
    LRU.addUnits(LRUs[MBB].getBitVector());
    // We return false to continue the recursion.
    return false;
  };

  /// \Returns the live-outs at \p Reload by starting with the block's live-outs
  /// and stepping backwards until we reach \p Reload.
  auto GetReloadLRU = [this](MachineInstr *Reload) {
    LiveRegUnits ReloadLRU(*TRI);
    MachineBasicBlock *MBB = Reload->getParent();
    ReloadLRU.addLiveOuts(*MBB);
    // Start at the bottom of the BB and walk up until we find `Reload`.
    for (MachineInstr &MI : llvm::reverse(*MBB)) {
      if (&MI == Reload)
        break;
      // We use stepBackward() instead of accumulate because we need to remove
      // the killed values from the live-outs.
      ReloadLRU.stepBackward(MI);
    }
    return ReloadLRU;
  };

  // Start from each Reload and walk up the CFG with a depth-first traversal,
  // looking for spills. Upon finding a spill we don't go beyond that point. In
  // the meantime we accumulate the registers used. This is then used to find
  // free physical registers.
  DenseSet<MachineBasicBlock *> Visited;
  for (const auto &ReloadData : Entry.Reloads) {
    MachineInstr *Reload = ReloadData.MI;
    // Add the Reload's LRU to the total LRU for the whole Spill-Reload range.
    //   bb:
    //    ...
    //    reload %stack.0 ^
    //    ...             | Compute live-outs at `reload` by stepping backwards
    //    ret             -
    LiveRegUnits ReloadLiveOuts = GetReloadLRU(Reload);
    // Now accumulate into `ReloadLiveOuts` by walking upwards from `Reload`
    // until we reach the first spill or the top of the BB.
    //   bb:
    //    ...
    //    spill %stack..0 ^
    //    ...             | Accumulate into ReloadLiveOuts
    //    reload %stack.0 -
    //    ...
    //    ret
    bool FoundSpill = AccumulateLRUUntilSpillFn(Reload, ReloadLiveOuts);
    LRU.addUnits(ReloadLiveOuts.getBitVector());

    // If we did not find a spill then we need to look for it. Traverse the CFG
    // bottom-up accumulating LRUs until we reach the Spills.
    if (!FoundSpill) {
      for (MachineBasicBlock *PredMBB : Reload->getParent()->predecessors())
        DFS(PredMBB, Visited, AccumulateLRUFn);
    }
  }
}

void Spill2Reg::generateCode() {
  for (auto &Pair : StackSlotData) {
    StackSlotDataEntry &Entry = Pair.second;
    // Skip if this stack slot was disabled during data collection.
    if (Entry.Disable)
      continue;

    // We decide to spill2reg if any of the spills/reloads are in a hotspot.
    if (!allAccessesProfitable(Entry))
      continue;

    // Calculate liveness for Entry.
    LiveRegUnits LRU(*TRI);
    calculateLiveRegs(Entry, LRU);

    // Look for a physical register that is not in LRU.
    std::optional<MCRegister> PhysVectorRegOpt = tryGetFreePhysicalReg(
        Entry.getSpilledRegClass(TII, TRI, &MF->getSubtarget()), LRU);
    if (!PhysVectorRegOpt)
      continue;

    // Update the MBB live-ins. These are used for the live regs calculation.
    // Collect the parent MBBs of Spills for fast lookup.
    // NOTE: We do that before calling replaceStackWithReg() because it will
    // remove the spill/reload instructions from Entry.
    updateLiveIns(Entry, *PhysVectorRegOpt);

    // Replace stack accesses with register accesses.
    replaceStackWithReg(Entry, *PhysVectorRegOpt);

    NumSpill2RegInstrs += Entry.Spills.size() + Entry.Reloads.size();
  }
}

void Spill2Reg::cleanup() {
  StackSlotData.clear();
  LRUs.clear();
}

bool Spill2Reg::run() {
  // Walk over each instruction in the code keeping track of the processor's
  // port pressure and look for memory unit hot-spots.
  collectSpillsAndReloads();

  // Replace each spills/reloads to stack slots with register spills/reloads.
  generateCode();

  cleanup();
  return true;
}

#ifndef NDEBUG
void Spill2Reg::StackSlotDataEntry::MIData::dump() const {
  dbgs() << "  (" << *MO << ") " << *MI;
}

void Spill2Reg::StackSlotDataEntry::MIDataWithLiveIn::dump() const {
  dbgs() << "  (" << *MO << ") " << *MI << " IsLiveIn: " << IsLiveIn;
}

void Spill2Reg::StackSlotDataEntry::dump() const {
  dbgs().indent(DumpInd) << "Disable: " << Disable << "\n";
  dbgs().indent(DumpInd) << "Spills:\n";
  for (const MIData &Data : Spills)
    Data.dump();
  dbgs().indent(DumpInd) << "Reloads:\n";
  for (const MIData &Data : Reloads)
    Data.dump();
}
#endif

INITIALIZE_PASS_BEGIN(Spill2Reg, "spill2reg", "Spill2Reg", false, false)
INITIALIZE_PASS_END(Spill2Reg, "spill2reg", "Spill2Reg", false, false)
