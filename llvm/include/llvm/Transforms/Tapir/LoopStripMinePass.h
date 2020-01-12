//===- LoopStripMinePass.h - Loop stripmining -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_LOOPSTRIPMINEPASS_H
#define LLVM_TRANSFORMS_TAPIR_LOOPSTRIPMINEPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

extern cl::opt<bool> EnableTapirLoopStripmine;

/// Loop stripmining pass.  It is a function pass to have access to function and
/// module analyses.
class LoopStripMinePass : public PassInfoMixin<LoopStripMinePass> {
public:
  explicit LoopStripMinePass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_LOOPSTRIPMINEPASS_H
