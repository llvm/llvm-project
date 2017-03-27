//===- llvm/CodeGen/GlobalISel/InstructionSelect.cpp - InstructionSelect ---==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the InstructionSelect class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#define DEBUG_TYPE "instruction-select"

using namespace llvm;

char InstructionSelect::ID = 0;
INITIALIZE_PASS_BEGIN(InstructionSelect, DEBUG_TYPE,
                      "Select target instructions out of generic instructions",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(InstructionSelect, DEBUG_TYPE,
                    "Select target instructions out of generic instructions",
                    false, false)

InstructionSelect::InstructionSelect() : MachineFunctionPass(ID) {
  initializeInstructionSelectPass(*PassRegistry::getPassRegistry());
}

void InstructionSelect::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool InstructionSelect::runOnMachineFunction(MachineFunction &MF) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  // No matter what happens, whether we successfully select the function or not,
  // nothing is going to use the vreg types after us.  Make sure they disappear.
  auto ClearVRegTypesOnReturn =
      make_scope_exit([&]() { MRI.getVRegToType().clear(); });

  // If the ISel pipeline failed, do not bother running that pass.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;

  DEBUG(dbgs() << "Selecting function: " << MF.getName() << '\n');

  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  const InstructionSelector *ISel = MF.getSubtarget().getInstructionSelector();
  assert(ISel && "Cannot work without InstructionSelector");

  // An optimization remark emitter. Used to report failures.
  MachineOptimizationRemarkEmitter MORE(MF, /*MBFI=*/nullptr);

  // FIXME: freezeReservedRegs is now done in IRTranslator, but there are many
  // other MF/MFI fields we need to initialize.

#ifndef NDEBUG
  // Check that our input is fully legal: we require the function to have the
  // Legalized property, so it should be.
  // FIXME: This should be in the MachineVerifier, but it can't use the
  // LegalizerInfo as it's currently in the separate GlobalISel library.
  // The RegBankSelected property is already checked in the verifier. Note
  // that it has the same layering problem, but we only use inline methods so
  // end up not needing to link against the GlobalISel library.
  if (const LegalizerInfo *MLI = MF.getSubtarget().getLegalizerInfo())
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : MBB)
        if (isPreISelGenericOpcode(MI.getOpcode()) && !MLI->isLegal(MI, MRI)) {
          reportGISelFailure(MF, TPC, MORE, "gisel-select",
                             "instruction is not legal", MI);
          return false;
        }

#endif
  // FIXME: We could introduce new blocks and will need to fix the outer loop.
  // Until then, keep track of the number of blocks to assert that we don't.
  const size_t NumBlocks = MF.size();

  for (MachineBasicBlock *MBB : post_order(&MF)) {
    if (MBB->empty())
      continue;

    // Select instructions in reverse block order. We permit erasing so have
    // to resort to manually iterating and recognizing the begin (rend) case.
    bool ReachedBegin = false;
    for (auto MII = std::prev(MBB->end()), Begin = MBB->begin();
         !ReachedBegin;) {
#ifndef NDEBUG
      // Keep track of the insertion range for debug printing.
      const auto AfterIt = std::next(MII);
#endif
      // Select this instruction.
      MachineInstr &MI = *MII;

      // And have our iterator point to the next instruction, if there is one.
      if (MII == Begin)
        ReachedBegin = true;
      else
        --MII;

      DEBUG(dbgs() << "Selecting: \n  " << MI);

      // We could have folded this instruction away already, making it dead.
      // If so, erase it.
      if (isTriviallyDead(MI, MRI)) {
        DEBUG(dbgs() << "Is dead; erasing.\n");
        MI.eraseFromParent();
        continue;
      }

      if (!ISel->select(MI)) {
        // FIXME: It would be nice to dump all inserted instructions.  It's
        // not obvious how, esp. considering select() can insert after MI.
        reportGISelFailure(MF, TPC, MORE, "gisel-select", "cannot select", MI);
        return false;
      }

      // Dump the range of instructions that MI expanded into.
      DEBUG({
        auto InsertedBegin = ReachedBegin ? MBB->begin() : std::next(MII);
        dbgs() << "Into:\n";
        for (auto &InsertedMI : make_range(InsertedBegin, AfterIt))
          dbgs() << "  " << InsertedMI;
        dbgs() << '\n';
      });
    }
  }

  // Now that selection is complete, there are no more generic vregs.  Verify
  // that the size of the now-constrained vreg is unchanged and that it has a
  // register class.
  for (auto &VRegToType : MRI.getVRegToType()) {
    unsigned VReg = VRegToType.first;
    auto *RC = MRI.getRegClassOrNull(VReg);
    MachineInstr *MI = nullptr;
    if (!MRI.def_empty(VReg))
      MI = &*MRI.def_instr_begin(VReg);
    else if (!MRI.use_empty(VReg))
      MI = &*MRI.use_instr_begin(VReg);

    if (MI && !RC) {
      reportGISelFailure(MF, TPC, MORE, "gisel-select",
                         "VReg has no regclass after selection", *MI);
      return false;
    } else if (!RC)
      continue;

    if (VRegToType.second.isValid() &&
        VRegToType.second.getSizeInBits() > (RC->getSize() * 8)) {
      reportGISelFailure(MF, TPC, MORE, "gisel-select",
                         "VReg has explicit size different from class size",
                         *MI);
      return false;
    }
  }

  if (MF.size() != NumBlocks) {
    MachineOptimizationRemarkMissed R("gisel-select", "GISelFailure",
                                      MF.getFunction()->getSubprogram(),
                                      /*MBB=*/nullptr);
    R << "inserting blocks is not supported yet";
    reportGISelFailure(MF, TPC, MORE, R);
    return false;
  }

  // FIXME: Should we accurately track changes?
  return true;
}
