//===- LoopInstSimplify.h - Loop Inst Simplify Pass -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs lightweight instruction simplification on loop bodies.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPINSTSIMPLIFY_H
#define LLVM_TRANSFORMS_SCALAR_LOOPINSTSIMPLIFY_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPassManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Performs Loop Inst Simplify Pass.
class LoopInstSimplifyPass : public PassInfoMixin<LoopInstSimplifyPass> {
public:
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPINSTSIMPLIFY_H
