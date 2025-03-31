//===----- InstRipplefpext.h - Update fpext inst to @llvm.ripple.fpext for bf16
//on Hexagon ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Clang does something beneficial for Hexagon by handling `bfloat16` codes as
// follows:
// 1. Promotes them (through `fpext`) to `fp32`.
// 2. Performs the computation in `fp32`.
// 3. Demotes them back to `bfloat16`.
// However, the InstCombine pass in LLVM "optimizes" this process by reverting
// everything back to `bfloat16` computations. To prevent this, we created a new
// LLVM intrinsic `@llvm.ripple.fpext` to replace `fpext`. This pass will
// execute before the first InstCombine pass to replace all `fpext` with
// `@llvm.ripple.fpext`.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_RIPPLE_FPEXT_FPTRUNC_H
#define LLVM_TRANSFORMS_VECTORIZE_RIPPLE_FPEXT_FPTRUNC_H

#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

class RippleFPExtFPTruncPass : public PassInfoMixin<RippleFPExtFPTruncPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_RIPPLE_FPEXT_FPTRUNC_H
