//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Try to increase the alignment of globals to at least 2, to avoid the need for
// GOT indirection to load an unaligned address.
//
//===----------------------------------------------------------------------===//

#include "SystemZ.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {

class SystemZAlignGlobals : public ModulePass {
public:
  static char ID;

  explicit SystemZAlignGlobals() : ModulePass(ID) {}

  StringRef getPassName() const override { return "SystemZ Align Globals"; }

  bool runOnModule(Module &M) override;
};

} // end anonymous namespace

char SystemZAlignGlobals::ID = 0;

INITIALIZE_PASS(SystemZAlignGlobals, "systemz-align-globals",
                "Try aligning globals to 2 bytes", false, false)

ModulePass *llvm::createSystemZAlignGlobalsPass() {
  return new SystemZAlignGlobals();
}

bool SystemZAlignGlobals::runOnModule(Module &M) {
  bool Changed = false;

  for (GlobalVariable &GV : M.globals()) {
    MaybeAlign GVAlign = GV.getAlign();
    if (!GVAlign || GVAlign.value() > 1)
      continue;

    if (!GV.canIncreaseAlignment())
      continue;

    GV.setAlignment(Align(2));
    Changed = true;
  }

  return Changed;
}
