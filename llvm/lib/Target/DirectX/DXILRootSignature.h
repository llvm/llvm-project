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
#ifndef LLVM_LIB_TARGET_DIRECTX_DXILROOTSIGNATURE_H
#define LLVM_LIB_TARGET_DIRECTX_DXILROOTSIGNATURE_H

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

class RootSignatureBindingInfo {
private:
  SmallDenseMap<const Function *, mcdxbc::RootSignatureDesc> FuncToRsMap;

public:
  using iterator =
      SmallDenseMap<const Function *, mcdxbc::RootSignatureDesc>::iterator;

  RootSignatureBindingInfo() = default;
  RootSignatureBindingInfo(
      SmallDenseMap<const Function *, mcdxbc::RootSignatureDesc> Map)
      : FuncToRsMap(Map) {};

  iterator find(const Function *F) { return FuncToRsMap.find(F); }

  iterator end() { return FuncToRsMap.end(); }

  mcdxbc::RootSignatureDesc *getDescForFunction(const Function *F) {
    const auto FuncRs = find(F);
    if (FuncRs == end())
      return nullptr;
    return &FuncRs->second;
  }
};

class RootSignatureAnalysis : public AnalysisInfoMixin<RootSignatureAnalysis> {
  friend AnalysisInfoMixin<RootSignatureAnalysis>;
  static AnalysisKey Key;

public:
  RootSignatureAnalysis() = default;

  using Result = RootSignatureBindingInfo;

  Result run(Module &M, ModuleAnalysisManager &AM);
};

/// Wrapper pass for the legacy pass manager.
///
/// This is required because the passes that will depend on this are codegen
/// passes which run through the legacy pass manager.
class RootSignatureAnalysisWrapper : public ModulePass {
private:
  std::unique_ptr<RootSignatureBindingInfo> FuncToRsMap;

public:
  static char ID;
  RootSignatureAnalysisWrapper() : ModulePass(ID) {}

  RootSignatureBindingInfo &getRSInfo() { return *FuncToRsMap; }

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
#endif
