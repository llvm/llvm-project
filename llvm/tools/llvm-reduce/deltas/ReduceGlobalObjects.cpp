//===- ReduceGlobalObjects.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReduceGlobalObjects.h"
#include "llvm/IR/GlobalObject.h"

using namespace llvm;

static bool shouldReduceSection(GlobalObject &GO) { return GO.hasSection(); }

static bool shouldReduceAlign(GlobalVariable *GV) {
  return GV->getAlign().has_value();
}

static bool shouldReduceAlign(Function *F) { return F->getAlign().has_value(); }

static bool shouldReduceComdat(GlobalObject &GO) { return GO.hasComdat(); }

void llvm::reduceGlobalObjectsDeltaPass(Oracle &O, ReducerWorkItem &Program) {
  for (auto &GO : Program.getModule().global_objects()) {
    if (shouldReduceSection(GO) && !O.shouldKeep())
      GO.setSection("");
    if (auto *GV = dyn_cast<GlobalVariable>(&GO)) {
      if (shouldReduceAlign(GV) && !O.shouldKeep())
        GV->setAlignment(MaybeAlign());
    }
    if (auto *F = dyn_cast<Function>(&GO)) {
      if (shouldReduceAlign(F) && !O.shouldKeep())
        F->setAlignment(MaybeAlign());
    }
    if (shouldReduceComdat(GO) && !O.shouldKeep())
      GO.setComdat(nullptr);
  }
}
