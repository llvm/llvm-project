//===------- Definition of the SanitizerBinaryMetadata class ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SanitizerBinaryMetadata pass.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_SANITIZERBINARYMETADATA_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_SANITIZERBINARYMETADATA_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Instrumentation.h"

namespace llvm {

/// Public interface to the SanitizerBinaryMetadata module pass for emitting
/// metadata for binary analysis sanitizers.
//
/// The pass should be inserted after optimizations.
class SanitizerBinaryMetadataPass
    : public PassInfoMixin<SanitizerBinaryMetadataPass> {
public:
  explicit SanitizerBinaryMetadataPass(
      SanitizerBinaryMetadataOptions Opts = {});
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  const SanitizerBinaryMetadataOptions Options;
};

} // namespace llvm

#endif
