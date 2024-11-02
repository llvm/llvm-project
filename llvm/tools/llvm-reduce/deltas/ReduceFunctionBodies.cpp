//===- ReduceFunctions.cpp - Specialized Delta Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce function bodies in the provided Module.
//
//===----------------------------------------------------------------------===//

#include "ReduceFunctionBodies.h"
#include "Delta.h"
#include "Utils.h"
#include "llvm/IR/GlobalValue.h"

using namespace llvm;

/// Removes all the bodies of defined functions that aren't inside any of the
/// desired Chunks.
static void extractFunctionBodiesFromModule(Oracle &O, Module &Program) {
  // Delete out-of-chunk function bodies
  for (auto &F : Program) {
    if (!F.isDeclaration() && !hasAliasUse(F) && !O.shouldKeep()) {
      F.deleteBody();
      F.setComdat(nullptr);
    }
  }
}

void llvm::reduceFunctionBodiesDeltaPass(TestRunner &Test) {
  runDeltaPass(Test, extractFunctionBodiesFromModule,
               "Reducing Function Bodies");
}
