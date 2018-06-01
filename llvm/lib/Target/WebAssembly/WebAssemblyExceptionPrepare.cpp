//=== WebAssemblyExceptionPrepare.cpp - WebAssembly Exception Preparation -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Does various transformations for exception handling.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-exception-prepare"

namespace {
class WebAssemblyExceptionPrepare final : public MachineFunctionPass {
  StringRef getPassName() const override {
    return "WebAssembly Prepare Exception";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  bool replaceFuncletReturns(MachineFunction &MF);

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyExceptionPrepare() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyExceptionPrepare::ID = 0;
INITIALIZE_PASS(WebAssemblyExceptionPrepare, DEBUG_TYPE,
                "WebAssembly Exception Preparation", false, false)

FunctionPass *llvm::createWebAssemblyExceptionPrepare() {
  return new WebAssemblyExceptionPrepare();
}

bool WebAssemblyExceptionPrepare::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  if (!MF.getFunction().hasPersonalityFn())
    return false;
  Changed |= replaceFuncletReturns(MF);
  // TODO More transformations will be added
  return Changed;
}

bool WebAssemblyExceptionPrepare::replaceFuncletReturns(MachineFunction &MF) {
  bool Changed = false;
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  for (auto &MBB : MF) {
    auto Pos = MBB.getFirstTerminator();
    if (Pos == MBB.end())
      continue;
    MachineInstr *TI = &*Pos;

    switch (TI->getOpcode()) {
    case WebAssembly::CATCHRET: {
      // Replace a catchret with a branch
      MachineBasicBlock *TBB = TI->getOperand(0).getMBB();
      if (!MBB.isLayoutSuccessor(TBB))
        BuildMI(MBB, TI, TI->getDebugLoc(), TII.get(WebAssembly::BR))
            .addMBB(TBB);
      TI->eraseFromParent();
      Changed = true;
      break;
    }
    case WebAssembly::CLEANUPRET: {
      // Replace a cleanupret with a rethrow
      BuildMI(MBB, TI, TI->getDebugLoc(), TII.get(WebAssembly::RETHROW))
          .addImm(0);
      TI->eraseFromParent();
      Changed = true;
      break;
    }
    }
  }
  return Changed;
}
