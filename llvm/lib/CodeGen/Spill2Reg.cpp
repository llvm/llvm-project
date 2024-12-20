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
      LLVM_DUMP_METHOD void dump() const;
#endif
    };
    SmallVector<MIData, 1> Spills;
    SmallVector<MIData, 1> Reloads;

    /// \Returns the register class of the register being spilled.
    const TargetRegisterClass *
    getSpilledRegClass(const TargetInstrInfo *TII,
                       const TargetRegisterInfo *TRI) const {
      auto Reg0 = Spills.front().MO->getReg();
      return TRI->getCandidateRegisterClassForSpill2Reg(TRI, Reg0);
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
    for (MachineInstr &MI : MBB) {
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
  }
}

bool Spill2Reg::isProfitable(const MachineInstr *MI) const {
  // TODO: Unimplemented.
  return true;
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

// Replace stack-based spills/reloads with register-based ones.
void Spill2Reg::replaceStackWithReg(StackSlotDataEntry &Entry,
                                    Register VectorReg) {
  // TODO: Unimplemented
}

void Spill2Reg::calculateLiveRegs(StackSlotDataEntry &Entry,
                                  LiveRegUnits &LRU) {
  // TODO: Unimplemented
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
    std::optional<MCRegister> PhysVectorRegOpt =
        tryGetFreePhysicalReg(Entry.getSpilledRegClass(TII, TRI), LRU);
    if (!PhysVectorRegOpt)
      continue;

    // Replace stack accesses with register accesses.
    replaceStackWithReg(Entry, *PhysVectorRegOpt);
  }
}

void Spill2Reg::cleanup() { StackSlotData.clear(); }

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
