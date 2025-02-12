//===- DXILRootSignature.h - DXIL Root Signature helper objects -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects and APIs for working with DXIL
///       Root Signatures.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {
namespace dxil {

enum class RootSignatureElementKind { None = 0, RootFlags = 1 };

struct ModuleRootSignature {
  ModuleRootSignature() = default;
  uint32_t Flags = 0;
};

class RootSignatureAnalysis : public AnalysisInfoMixin<RootSignatureAnalysis> {
  friend AnalysisInfoMixin<RootSignatureAnalysis>;
  static AnalysisKey Key;

public:
  RootSignatureAnalysis() = default;

  using Result = SmallDenseMap<const Function *, ModuleRootSignature>;

  SmallDenseMap<const Function *, ModuleRootSignature>
  run(Module &M, ModuleAnalysisManager &AM);
};

/// Wrapper pass for the legacy pass manager.
///
/// This is required because the passes that will depend on this are codegen
/// passes which run through the legacy pass manager.
class RootSignatureAnalysisWrapper : public ModulePass {
private:
  SmallDenseMap<const Function *, ModuleRootSignature> MRS;

public:
  static char ID;

  RootSignatureAnalysisWrapper() : ModulePass(ID) {}

  bool hasForFunction(const Function *F) { return MRS.find(F) != MRS.end(); }

  ModuleRootSignature getForFunction(const Function *F) {
    assert(hasForFunction(F));
    return MRS[F];
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

/// Printer pass for RootSignatureAnalysis results.
class RootSignatureAnalysisPrinter
    : public PassInfoMixin<RootSignatureAnalysisPrinter> {
  raw_ostream &OS;

public:
  explicit RootSignatureAnalysisPrinter(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace dxil
} // namespace llvm
