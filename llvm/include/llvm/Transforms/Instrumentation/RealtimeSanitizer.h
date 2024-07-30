//===--------- Definition of the RealtimeSanitizer class ---------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_REALTIMESANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_REALTIMESANITIZER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Instrumentation/RealtimeSanitizerOptions.h"

namespace llvm {

struct RealtimeSanitizerOptions {};

class RealtimeSanitizerPass : public PassInfoMixin<RealtimeSanitizerPass> {
public:
  RealtimeSanitizerPass(const RealtimeSanitizerOptions &Options);
  PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);

  static bool isRequired() { return true; }

private:
  RealtimeSanitizerOptions Options{};
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_REALTIMESANITIZER_H
