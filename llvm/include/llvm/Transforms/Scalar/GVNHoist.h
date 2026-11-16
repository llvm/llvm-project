//===- GVNHoist.h - Hoist scalar and load expressions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the interface for the GVNHoist pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_GVNHOIST_H
#define LLVM_TRANSFORMS_SCALAR_GVNHOIST_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// A simple and fast domtree-based GVN pass to hoist common expressions
/// from sibling branches.
struct GVNHoistPass : OptionalPassInfoMixin<GVNHoistPass> {
  /// Run the pass over the function.
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_GVNHOIST_H
