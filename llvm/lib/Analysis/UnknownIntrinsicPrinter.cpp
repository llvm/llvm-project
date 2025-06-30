//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/UnknownIntrinsicPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

using namespace llvm;

PreservedAnalyses UnknownIntrinsicPrinterPass::run(Module &M,
                                                   ModuleAnalysisManager &) {
  for (const Function &F : M.functions()) {
    // An unknown intrinsic is a function that begins with "llvm." but is not
    // a recognized intrinsic (i.e., it's IntrinsicID is not_intrinsic).
    if (F.isIntrinsic() && F.getIntrinsicID() == Intrinsic::not_intrinsic)
      OS << "Unknown intrinsic : " << F.getName() << '\n';
  }
  return PreservedAnalyses::all();
}
