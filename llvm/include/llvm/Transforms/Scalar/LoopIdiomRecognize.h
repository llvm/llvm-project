//===- LoopIdiomRecognize.h - Loop Idiom Recognize Pass -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements an idiom recognizer that transforms simple loops into a
// non-loop form.  In cases that this kicks in, it can be a significant
// performance win.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPIDIOMRECOGNIZE_H
#define LLVM_TRANSFORMS_SCALAR_LOOPIDIOMRECOGNIZE_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Performs Loop Idiom Recognize Pass.
class LoopIdiomRecognizePass : public PassInfoMixin<LoopIdiomRecognizePass> {
public:
  PreservedAnalyses run(Loop &L, AnalysisManager<Loop> &AM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPIDIOMRECOGNIZE_H
