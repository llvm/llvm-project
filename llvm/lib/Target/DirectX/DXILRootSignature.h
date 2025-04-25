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
#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/Pass.h"
#include <optional>

namespace llvm {
namespace dxil {

enum class RootSignatureElementKind {
  Error = 0,
  RootFlags = 1,
  RootConstants = 2
};
class RootSignatureAnalysis : public AnalysisInfoMixin<RootSignatureAnalysis> {
  friend AnalysisInfoMixin<RootSignatureAnalysis>;
  static AnalysisKey Key;

public:
  RootSignatureAnalysis() = default;

  using Result = SmallDenseMap<const Function *, mcdxbc::RootSignatureDesc>;

  SmallDenseMap<const Function *, mcdxbc::RootSignatureDesc>
  run(Module &M, ModuleAnalysisManager &AM);
};

/// Wrapper pass for the legacy pass manager.
///
/// This is required because the passes that will depend on this are codegen
/// passes which run through the legacy pass manager.
class RootSignatureAnalysisWrapper : public ModulePass {
private:
  SmallDenseMap<const Function *, mcdxbc::RootSignatureDesc> FuncToRsMap;

public:
  static char ID;

  RootSignatureAnalysisWrapper() : ModulePass(ID) {}

  using iterator =
      SmallDenseMap<const Function *, mcdxbc::RootSignatureDesc>::iterator;

  iterator find(const Function *F) { return FuncToRsMap.find(F); }

  iterator end() { return FuncToRsMap.end(); }

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
