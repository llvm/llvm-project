//===-- llvm/CodeGen/GlobalISel/Legalizer.cpp -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file implements the LegalizerHelper class to legalize individual
/// instructions and the LegalizePass wrapper pass for the primary
/// legalization.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/GlobalISel/LegalizerCombiner.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#include <iterator>

#define DEBUG_TYPE "legalizer"

using namespace llvm;

char Legalizer::ID = 0;
INITIALIZE_PASS_BEGIN(Legalizer, DEBUG_TYPE,
                      "Legalize the Machine IR a function's Machine IR", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(Legalizer, DEBUG_TYPE,
                    "Legalize the Machine IR a function's Machine IR", false,
                    false)

Legalizer::Legalizer() : MachineFunctionPass(ID) {
  initializeLegalizerPass(*PassRegistry::getPassRegistry());
}

void Legalizer::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void Legalizer::init(MachineFunction &MF) {
}

bool Legalizer::runOnMachineFunction(MachineFunction &MF) {
  // If the ISel pipeline failed, do not bother running that pass.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  DEBUG(dbgs() << "Legalize Machine IR for: " << MF.getName() << '\n');
  init(MF);
  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  MachineOptimizationRemarkEmitter MORE(MF, /*MBFI=*/nullptr);
  LegalizerHelper Helper(MF);

  // FIXME: an instruction may need more than one pass before it is legal. For
  // example on most architectures <3 x i3> is doubly-illegal. It would
  // typically proceed along a path like: <3 x i3> -> <3 x i8> -> <8 x i8>. We
  // probably want a worklist of instructions rather than naive iterate until
  // convergence for performance reasons.
  bool Changed = false;
  MachineBasicBlock::iterator NextMI;
  for (auto &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); MI = NextMI) {
      // Get the next Instruction before we try to legalize, because there's a
      // good chance MI will be deleted.
      NextMI = std::next(MI);

      // Only legalize pre-isel generic instructions: others don't have types
      // and are assumed to be legal.
      if (!isPreISelGenericOpcode(MI->getOpcode()))
        continue;
      unsigned NumNewInsns = 0;
      using VecType = SetVector<MachineInstr *, SmallVector<MachineInstr *, 8>>;
      VecType WorkList;
      VecType CombineList;
      Helper.MIRBuilder.recordInsertions([&](MachineInstr *MI) {
        // Only legalize pre-isel generic instructions.
        // Legalization process could generate Target specific pseudo
        // instructions with generic types. Don't record them
        if (isPreISelGenericOpcode(MI->getOpcode())) {
          ++NumNewInsns;
          WorkList.insert(MI);
          CombineList.insert(MI);
        }
      });
      WorkList.insert(&*MI);
      LegalizerCombiner C(Helper.MIRBuilder, MF.getRegInfo());
      bool Changed = false;
      LegalizerHelper::LegalizeResult Res;
      do {
        assert(!WorkList.empty() && "Expecting illegal ops");
        while (!WorkList.empty()) {
          NumNewInsns = 0;
          MachineInstr *CurrInst = WorkList.pop_back_val();
          Res = Helper.legalizeInstrStep(*CurrInst);
          // Error out if we couldn't legalize this instruction. We may want to
          // fall back to DAG ISel instead in the future.
          if (Res == LegalizerHelper::UnableToLegalize) {
            Helper.MIRBuilder.stopRecordingInsertions();
            if (Res == LegalizerHelper::UnableToLegalize) {
              reportGISelFailure(MF, TPC, MORE, "gisel-legalize",
                                 "unable to legalize instruction", *CurrInst);
              return false;
            }
          }
          Changed |= Res == LegalizerHelper::Legalized;
          // If CurrInst was legalized, there's a good chance that it might have
          // been erased. So remove it from the Combine List.
          if (Res == LegalizerHelper::Legalized)
            CombineList.remove(CurrInst);

#ifndef NDEBUG
          if (NumNewInsns)
            for (unsigned I = WorkList.size() - NumNewInsns,
                          E = WorkList.size();
                 I != E; ++I)
              DEBUG(dbgs() << ".. .. New MI: " << *WorkList[I];);
#endif
        }
        // Do the combines.
        while (!CombineList.empty()) {
          NumNewInsns = 0;
          MachineInstr *CurrInst = CombineList.pop_back_val();
          SmallVector<MachineInstr *, 4> DeadInstructions;
          Changed |= C.tryCombineInstruction(*CurrInst, DeadInstructions);
          for (auto *DeadMI : DeadInstructions) {
            DEBUG(dbgs() << ".. Erasing Dead Instruction " << *DeadMI);
            CombineList.remove(DeadMI);
            WorkList.remove(DeadMI);
            DeadMI->eraseFromParent();
          }
#ifndef NDEBUG
          if (NumNewInsns)
            for (unsigned I = CombineList.size() - NumNewInsns,
                          E = CombineList.size();
                 I != E; ++I)
              DEBUG(dbgs() << ".. .. Combine New MI: " << *CombineList[I];);
#endif
        }
      } while (!WorkList.empty());

      Helper.MIRBuilder.stopRecordingInsertions();
    }
  }

  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineIRBuilder MIRBuilder(MF);
  LegalizerCombiner C(MIRBuilder, MRI);
  for (auto &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); MI = NextMI) {
      // Get the next Instruction before we try to legalize, because there's a
      // good chance MI will be deleted.
      // TOOD: Perhaps move this to a combiner pass later?.
      NextMI = std::next(MI);
      SmallVector<MachineInstr *, 4> DeadInsts;
      Changed |= C.tryCombineMerges(*MI, DeadInsts);
      for (auto *DeadMI : DeadInsts)
        DeadMI->eraseFromParent();
    }
  }

  return Changed;
}
