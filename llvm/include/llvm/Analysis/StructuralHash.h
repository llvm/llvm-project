//=- StructuralHash.h - Structural Hash Printing and Analysis --*- C++ -*----=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_STRUCTURALHASH_H
#define LLVM_ANALYSIS_STRUCTURALHASH_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/StructuralHash.h"

namespace llvm {

struct PreservedFunctionHashAnalysis
    : public AnalysisInfoMixin<PreservedFunctionHashAnalysis> {
  static AnalysisKey Key;

  struct FunctionHash {
    uint64_t Hash;
  };

  using Result = FunctionHash;

  Result run(Function &F, FunctionAnalysisManager &FAM) {
    return Result{StructuralHash(F)};
  }
};

struct PreservedModuleHashAnalysis
    : public AnalysisInfoMixin<PreservedModuleHashAnalysis> {
  static AnalysisKey Key;

  struct ModuleHash {
    uint64_t Hash;
  };

  using Result = ModuleHash;

  Result run(Module &F, ModuleAnalysisManager &FAM) {
    return Result{StructuralHash(F)};
  }
};

enum class StructuralHashOptions {
  None,              /// Hash with opcode only.
  Detailed,          /// Hash with opcode and operands.
  CallTargetIgnored, /// Ignore call target operand when computing hash.
};

/// Printer pass for  StructuralHashes
class StructuralHashPrinterPass
    : public PassInfoMixin<StructuralHashPrinterPass> {
  raw_ostream &OS;
  const StructuralHashOptions Options;

public:
  explicit StructuralHashPrinterPass(raw_ostream &OS,
                                     StructuralHashOptions Options)
      : OS(OS), Options(Options) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_ANALYSIS_STRUCTURALHASH_H
