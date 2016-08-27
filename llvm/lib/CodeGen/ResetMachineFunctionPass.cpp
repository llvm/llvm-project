//===-- ResetMachineFunctionPass.cpp - Machine Loop Invariant Code Motion Pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "reset-machine-function"

namespace {
  class ResetMachineFunction : public MachineFunctionPass {
  public:
    static char ID; // Pass identification, replacement for typeid
    ResetMachineFunction() :
      MachineFunctionPass(ID) {
    }

    const char *getPassName() const override {
      return "ResetMachineFunction";
    }

    bool runOnMachineFunction(MachineFunction &MF) override {
      if (MF.getProperties().hasProperty(
              MachineFunctionProperties::Property::FailedISel)) {
        DEBUG(dbgs() << "Reseting: " << MF.getName() << '\n');
        MF.reset();
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
llvm::createResetMachineFunctionPass() {
  return new ResetMachineFunction();
}
