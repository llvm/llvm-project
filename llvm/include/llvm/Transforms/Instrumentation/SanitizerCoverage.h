//===--------- Definition of the SanitizerCoverage class --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SanitizerCoverage class which is a port of the legacy
// SanitizerCoverage pass to use the new PassManager infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_SANITIZERCOVERAGE_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_SANITIZERCOVERAGE_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Instrumentation.h"

namespace llvm {

/// This is the ModuleSanitizerCoverage pass used in the new pass manager. The
/// pass instruments functions for coverage, adds initialization calls to the
/// module for trace PC guards and 8bit counters if they are requested, and
/// appends globals to llvm.compiler.used.
class ModuleSanitizerCoveragePass
    : public PassInfoMixin<ModuleSanitizerCoveragePass> {
public:
  explicit ModuleSanitizerCoveragePass(
      SanitizerCoverageOptions Options = SanitizerCoverageOptions())
      : Options(Options) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  SanitizerCoverageOptions Options;
};

// Insert SanitizerCoverage instrumentation.
ModulePass *createModuleSanitizerCoverageLegacyPassPass(
    const SanitizerCoverageOptions &Options = SanitizerCoverageOptions());

} // namespace llvm

#endif
