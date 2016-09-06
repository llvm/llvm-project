//===- LoopUnrollPass.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPUNROLLPASS_H
#define LLVM_TRANSFORMS_SCALAR_LOOPUNROLLPASS_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPassManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

struct LoopUnrollPass : public PassInfoMixin<LoopUnrollPass> {
  Optional<unsigned> ProvidedCount;
  Optional<unsigned> ProvidedThreshold;
  Optional<bool> ProvidedAllowPartial;
  Optional<bool> ProvidedRuntime;

  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPUNROLLPASS_H
