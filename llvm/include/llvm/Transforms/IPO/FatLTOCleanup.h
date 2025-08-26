//===- FatLtoCleanup.h - clean up IR for the FatLTO pipeline ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines operations used to clean up IR for the FatLTO pipeline.
// Instrumentation that is beneficial for bitcode sections used in LTO may
// need to be cleaned up to finish non-LTO compilation. llvm.checked.load is
// an example of an instruction that we want to preserve for LTO, but is
// incorrect to leave unchanged during the per-TU compilation in FatLTO.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_FATLTOCLEANUP_H
#define LLVM_TRANSFORMS_IPO_FATLTOCLEANUP_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;
class ModuleSummaryIndex;

class FatLtoCleanup : public PassInfoMixin<FatLtoCleanup> {
public:
  FatLtoCleanup() {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_FATLTOCLEANUP_H
