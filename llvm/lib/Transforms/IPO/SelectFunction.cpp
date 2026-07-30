//===-- SelectFunction.cpp - Compile only a selected function -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/SelectFunction.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "select-function"

PreservedAnalyses SelectFunctionPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  StringSet<> Roots;
  for (const auto &Name : FunctionNames) {
    Function *F = M.getFunction(Name);
    if (!F || F->isDeclaration()) {
      errs() << "select-function: function '" << Name
             << "' not found in module\n";
      return PreservedAnalyses::all();
    }
    Roots.insert(Name);
  }

  auto MustPreserve = [&](const GlobalValue &GV) {
    return Roots.count(GV.getName());
  };
  InternalizePass Internalizer(MustPreserve);
  Internalizer.run(M, AM);

  GlobalDCEPass DCE;
  DCE.run(M, AM);

  StripDeadPrototypesPass StripProtos;
  StripProtos.run(M, AM);

  return PreservedAnalyses::none();
}
