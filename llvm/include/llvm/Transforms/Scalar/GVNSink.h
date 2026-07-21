//===- GVNSink.h - Sink expressions into successors -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the interface for the GVNSink pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_GVNSINK_H
#define LLVM_TRANSFORMS_SCALAR_GVNSINK_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Uses an "inverted" value numbering to decide the similarity of
/// expressions and sinks similar expressions into successors.
struct GVNSinkPass : OptionalPassInfoMixin<GVNSinkPass> {
  /// Run the pass over the function.
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_GVNSINK_H
