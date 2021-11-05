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

static bool shouldReduceAlign(GlobalObject &GO) {
  return GO.getAlign().hasValue();
}

static void reduceGOs(Oracle &O, Module &Program) {
  for (auto &GO : Program.global_objects()) {
    if (shouldReduceSection(GO) && !O.shouldKeep())
      GO.setSection("");
    if (shouldReduceAlign(GO) && !O.shouldKeep())
      GO.setAlignment(MaybeAlign());
  }
}

static int countGOs(Module &Program) {
  int SectionCount = count_if(Program.global_objects(), [](GlobalObject &GO) {
    return shouldReduceSection(GO);
  });
  int AlignCount = count_if(Program.global_objects(), [](GlobalObject &GO) {
    return shouldReduceAlign(GO);
  });
  return SectionCount + AlignCount;
}

void llvm::reduceGlobalObjectsDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing GlobalObjects...\n";
  int GVCount = countGOs(Test.getProgram());
  runDeltaPass(Test, GVCount, reduceGOs);
}
