//===- DXILRootSignature.h - DXIL Root Signature helper objects
//---------------===//
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

#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include <optional>

namespace llvm {
namespace dxil {

enum class RootSignatureElementKind {
  None = 0,
  RootFlags = 1,
  RootConstants = 2,
  RootDescriptor = 3,
  DescriptorTable = 4,
  StaticSampler = 5
};

struct ModuleRootSignature {
  uint32_t Flags;

  ModuleRootSignature() = default;

  bool parse(NamedMDNode *Root);

  static ModuleRootSignature analyzeModule(Module &M);
};

class RootSignatureAnalysis : public AnalysisInfoMixin<RootSignatureAnalysis> {
  friend AnalysisInfoMixin<RootSignatureAnalysis>;
  static AnalysisKey Key;

public:
  RootSignatureAnalysis() = default;

  using Result = ModuleRootSignature;

  ModuleRootSignature run(Module &M, ModuleAnalysisManager &AM);
};

/// Wrapper pass for the legacy pass manager.
///
/// This is required because the passes that will depend on this are codegen
/// passes which run through the legacy pass manager.
class RootSignatureAnalysisWrapper : public ModulePass {
  std::optional<ModuleRootSignature> MRS;

public:
  static char ID;

  RootSignatureAnalysisWrapper() : ModulePass(ID) {}

  const std::optional<ModuleRootSignature> &getRootSignature() { return MRS; }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // namespace dxil
} // namespace llvm
