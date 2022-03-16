//=== WebAssemblyNullifyDebugValueLists.cpp - Nullify DBG_VALUE_LISTs   ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Nullify DBG_VALUE_LISTs instructions as a temporary measure before we
/// implement DBG_VALUE_LIST handling in WebAssemblyDebugValueManager.
/// See https://bugs.llvm.org/show_bug.cgi?id=50361.
/// TODO Correctly handle DBG_VALUE_LISTs
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-nullify-dbg-value-lists"

namespace {
class WebAssemblyNullifyDebugValueLists final : public MachineFunctionPass {
  StringRef getPassName() const override {
    return "WebAssembly Nullify DBG_VALUE_LISTs";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyNullifyDebugValueLists() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyNullifyDebugValueLists::ID = 0;
INITIALIZE_PASS(WebAssemblyNullifyDebugValueLists, DEBUG_TYPE,
                "WebAssembly Nullify DBG_VALUE_LISTs", false, false)

FunctionPass *llvm::createWebAssemblyNullifyDebugValueLists() {
  return new WebAssemblyNullifyDebugValueLists();
}

bool WebAssemblyNullifyDebugValueLists::runOnMachineFunction(
    MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** Nullify DBG_VALUE_LISTs **********\n"
                       "********** Function: "
                    << MF.getName() << '\n');
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  SmallVector<MachineInstr *, 2> DbgValueLists;
  for (auto &MBB : MF)
    for (auto &MI : MBB)
      if (MI.getOpcode() == TargetOpcode::DBG_VALUE_LIST)
        DbgValueLists.push_back(&MI);

  // Our backend, including WebAssemblyDebugValueManager, currently cannot
  // handle DBG_VALUE_LISTs correctly. So this converts DBG_VALUE_LISTs to
  // "DBG_VALUE $noreg", which will appear as "optimized out".
  for (auto *DVL : DbgValueLists) {
    BuildMI(*DVL->getParent(), DVL, DVL->getDebugLoc(),
            TII.get(TargetOpcode::DBG_VALUE), false, Register(),
            DVL->getOperand(0).getMetadata(), DVL->getOperand(1).getMetadata());
    DVL->eraseFromParent();
  }

  return !DbgValueLists.empty();
}
