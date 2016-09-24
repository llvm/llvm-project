//===-- ResetMachineFunctionPass.cpp - Reset Machine Function ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a pass that will conditionally reset a machine
/// function as if it was just created. This is used to provide a fallback
/// mechanism when GlobalISel fails, thus the condition for the reset to
/// happen is that the MachineFunction has the FailedISel property.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "reset-machine-function"

STATISTIC(NumFunctionsReset, "Number of functions reset");

namespace {
  class ResetMachineFunction : public MachineFunctionPass {
    /// Tells whether or not this pass should emit a fallback
    /// diagnostic when it resets a function.
    bool EmitFallbackDiag;

  public:
    static char ID; // Pass identification, replacement for typeid
    ResetMachineFunction(bool EmitFallbackDiag = false)
        : MachineFunctionPass(ID), EmitFallbackDiag(EmitFallbackDiag) {}

    const char *getPassName() const override {
      return "ResetMachineFunction";
    }

    bool runOnMachineFunction(MachineFunction &MF) override {
      if (MF.getProperties().hasProperty(
              MachineFunctionProperties::Property::FailedISel)) {
        DEBUG(dbgs() << "Reseting: " << MF.getName() << '\n');
        ++NumFunctionsReset;
        MF.reset();
        if (EmitFallbackDiag) {
          const Function &F = *MF.getFunction();
          DiagnosticInfoISelFallback DiagFallback(F);
          F.getContext().diagnose(DiagFallback);
        }
        return true;
      }
      return false;
    }

  };
} // end anonymous namespace

char ResetMachineFunction::ID = 0;
INITIALIZE_PASS(ResetMachineFunction, DEBUG_TYPE,
                "reset machine function if ISel failed", false, false)

MachineFunctionPass *
llvm::createResetMachineFunctionPass(bool EmitFallbackDiag = false) {
  return new ResetMachineFunction(EmitFallbackDiag);
}
