//===----- ExpandLargeDivRem.h - Expand large div/rem ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_EXPANDLARGEDIVREM_H
#define LLVM_CODEGEN_EXPANDLARGEDIVREM_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Expands div/rem instructions with a bitwidth above a threshold
/// into a loop.
/// This is useful for backends like x86 that cannot lower divisions
/// with more than 128 bits.
class ExpandLargeDivRemPass : public PassInfoMixin<ExpandLargeDivRemPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  // The backend asserts when seeing large div/rem instructions.
  static bool isRequired() { return true; }
};
} // end namespace llvm

#endif // LLVM_CODEGEN_EXPANDLARGEDIVREM_H
