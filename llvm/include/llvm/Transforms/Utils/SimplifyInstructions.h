//===- SimplifyInstructions.h - Remove redundant instructions ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility pass used for testing the InstructionSimplify analysis.
// The analysis is applied to every instruction, and if it simplifies then the
// instruction is replaced by the simplification.  If you are looking for a pass
// that performs serious instruction folding, use the instcombine pass instead.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SIMPLIFYINSTRUCTIONS_H
#define LLVM_TRANSFORMS_UTILS_SIMPLIFYINSTRUCTIONS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// This pass removes redundant instructions.
class InstSimplifierPass : public PassInfoMixin<InstSimplifierPass> {
public:
  PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SIMPLIFYINSTRUCTIONS_H
