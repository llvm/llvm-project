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

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
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
  uint32_t Flags = 0;
  ModuleRootSignature() { Ctx = nullptr; };
  static std::optional<ModuleRootSignature> analyzeModule(Module &M,
                                                          const Function *F);

private:
  LLVMContext *Ctx;

  ModuleRootSignature(LLVMContext *Ctx) : Ctx(Ctx) {}

  bool parse(NamedMDNode *Root, const Function *F);
  bool parseRootSignatureElement(MDNode *Element);
  bool parseRootFlags(MDNode *RootFlagNode);

  bool validate();

  bool reportError(Twine Message, DiagnosticSeverity Severity = DS_Error);
};

using OptionalRootSignature = std::optional<ModuleRootSignature>;

class RootSignatureAnalysis : public AnalysisInfoMixin<RootSignatureAnalysis> {
  friend AnalysisInfoMixin<RootSignatureAnalysis>;
  static AnalysisKey Key;

public:
  RootSignatureAnalysis() = default;

  using Result = OptionalRootSignature;

  OptionalRootSignature run(Module &M, ModuleAnalysisManager &AM);
};

/// Wrapper pass for the legacy pass manager.
///
/// This is required because the passes that will depend on this are codegen
/// passes which run through the legacy pass manager.
class RootSignatureAnalysisWrapper : public ModulePass {
private:
  OptionalRootSignature MRS;

public:
  static char ID;

  RootSignatureAnalysisWrapper() : ModulePass(ID) {}

  const ModuleRootSignature &getRootSignature() { return MRS.value(); }

  bool hasRootSignature() { return MRS.has_value(); }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // namespace dxil
} // namespace llvm
