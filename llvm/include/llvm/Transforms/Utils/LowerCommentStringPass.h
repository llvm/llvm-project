//===- LowerCommentStringPass.h - Lower loadtime comments -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOWERCOMMENTSTRINGPASS_H
#define LLVM_TRANSFORMS_UTILS_LOWERCOMMENTSTRINGPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {
/// Attach !implicit.ref metadata from every defined function to each global
/// carrying !loadtime_comment metadata, so that the backend emits a reference
/// that keeps the loadtime identifying string through linking. See the
/// implementation file for the producers of such globals.
class LowerCommentStringPass
    : public RequiredPassInfoMixin<LowerCommentStringPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LOWERCOMMENTSTRINGPASS_H
