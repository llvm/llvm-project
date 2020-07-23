//===- AggressiveInstCombine.h - AggressiveInstCombine pass -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides the primary interface to the aggressive instcombine pass.
/// This pass is suitable for use in the new pass manager. For a pass that works
/// with the legacy pass manager, please use
/// \c createAggressiveInstCombinerPass().
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_AGGRESSIVE_INSTCOMBINE_INSTCOMBINE_H
#define LLVM_TRANSFORMS_AGGRESSIVE_INSTCOMBINE_INSTCOMBINE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class AggressiveInstCombinePass
    : public PassInfoMixin<AggressiveInstCombinePass> {

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

//===----------------------------------------------------------------------===//
//
// AggressiveInstCombiner - Combine expression patterns to form expressions with
// fewer, simple instructions. This pass does not modify the CFG.
//
FunctionPass *createAggressiveInstCombinerPass();
}

#endif
