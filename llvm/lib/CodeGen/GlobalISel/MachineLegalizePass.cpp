//===-- llvm/CodeGen/GlobalISel/MachineLegalizePass.cpp -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file implements the LegalizeHelper class to legalize individual
/// instructions and the MachineLegalizePass wrapper pass for the primary
/// legalization.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/MachineLegalizePass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineLegalizeHelper.h"
#include "llvm/CodeGen/GlobalISel/MachineLegalizer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#define DEBUG_TYPE "legalize-mir"

using namespace llvm;

char MachineLegalizePass::ID = 0;
INITIALIZE_PASS(MachineLegalizePass, DEBUG_TYPE,
                "Legalize the Machine IR a function's Machine IR", false,
                false)

MachineLegalizePass::MachineLegalizePass() : MachineFunctionPass(ID) {
  initializeMachineLegalizePassPass(*PassRegistry::getPassRegistry());
}

void MachineLegalizePass::init(MachineFunction &MF) {
}

bool MachineLegalizePass::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "Legalize Machine IR for: " << MF.getName() << '\n');
  init(MF);
  const MachineLegalizer &Legalizer = *MF.getSubtarget().getMachineLegalizer();
  MachineLegalizeHelper Helper(MF);

  // FIXME: an instruction may need more than one pass before it is legal. For
  // example on most architectures <3 x i3> is doubly-illegal. It would
  // typically proceed along a path like: <3 x i3> -> <3 x i8> -> <8 x i8>. We
  // probably want a worklist of instructions rather than naive iterate until
  // convergence for performance reasons.
  bool Changed = false;
  MachineBasicBlock::iterator NextMI;
  for (auto &MBB : MF)
    for (auto MI = MBB.begin(); MI != MBB.end(); MI = NextMI) {
      // Get the next Instruction before we try to legalize, because there's a
      // good chance MI will be deleted.
      NextMI = std::next(MI);

      // Only legalize pre-isel generic instructions: others don't have types
      // and are assumed to be legal.
      if (!isPreISelGenericOpcode(MI->getOpcode()))
        continue;

      auto Res = Helper.legalizeInstr(*MI, Legalizer);

      // Error out if we couldn't legalize this instruction. We may want to fall
      // back to DAG ISel instead in the future.
      if (Res == MachineLegalizeHelper::UnableToLegalize) {
        std::string Msg;
        raw_string_ostream OS(Msg);
        OS << "unable to legalize instruction: ";
        MI->print(OS);
        report_fatal_error(OS.str());
      }

      Changed |= Res == MachineLegalizeHelper::Legalized;
    }
  return Changed;
}
