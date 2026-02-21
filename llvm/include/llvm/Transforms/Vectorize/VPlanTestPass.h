//===- VPlanTestPass.h - Test VPlan transforms ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is a lightweight testing harness for VPlan transforms. It builds
// VPlan0 for loops and runs specified transforms.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANTESTPASS_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANTESTPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class VPlanTestPass : public PassInfoMixin<VPlanTestPass> {
  StringRef TransformPipeline;

public:
  VPlanTestPass(StringRef Pipeline = "") : TransformPipeline(Pipeline) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANTESTPASS_H
