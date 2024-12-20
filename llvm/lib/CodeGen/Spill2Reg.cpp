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

#ifndef NDEBUG
    LLVM_DUMP_METHOD void dump() const;
#endif
  };
  /// Look for candidates for spill2reg. These candidates are in places with
  /// high memory unit contention. Fills in StackSlotData.
  void collectSpillsAndReloads();
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

void Spill2Reg::generateCode() { llvm_unreachable("Unimplemented"); }

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
