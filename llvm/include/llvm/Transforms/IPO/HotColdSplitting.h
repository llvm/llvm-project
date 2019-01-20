//===- HotColdSplitting.h ---- Outline Cold Regions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass outlines cold regions to a separate function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_HOTCOLDSPLITTING_H
#define LLVM_TRANSFORMS_IPO_HOTCOLDSPLITTING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

/// Pass to outline cold regions.
class HotColdSplittingPass : public PassInfoMixin<HotColdSplittingPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_HOTCOLDSPLITTING_H

