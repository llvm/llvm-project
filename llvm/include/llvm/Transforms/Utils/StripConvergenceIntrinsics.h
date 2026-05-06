//===- StripConvergenceIntrinsics.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This pass strips convergence intrinsics and operand bundles as those are
/// only useful when modifying the CFG during IR passes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_STRIPCONVERGENCEINTRINSICS_H
#define LLVM_TRANSFORMS_UTILS_STRIPCONVERGENCEINTRINSICS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class StripConvergenceIntrinsicsPass
    : public OptionalPassInfoMixin<StripConvergenceIntrinsicsPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_STRIPCONVERGENCEINTRINSICS_H
