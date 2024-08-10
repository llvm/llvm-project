//===- RealtimeSanitizer.h - RealtimeSanitizer instrumentation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the RealtimeSanitizer, an LLVM transformation for
// detecting and reporting realtime safety violations.
//
// The instrumentation pass inserts calls to __rtsan_realtime_enter and
// __rtsan_realtime_exit at the entry and exit points of functions that are
// marked with the appropriate attribute.
//
// See also: llvm-project/compiler-rt/lib/rtsan/
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_REALTIMESANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_REALTIMESANITIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

struct RealtimeSanitizerOptions {};

class RealtimeSanitizerPass : public PassInfoMixin<RealtimeSanitizerPass> {
public:
  RealtimeSanitizerPass(const RealtimeSanitizerOptions &Options);
  PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_REALTIMESANITIZER_H
