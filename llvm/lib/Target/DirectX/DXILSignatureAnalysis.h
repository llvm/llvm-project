//===- DXILSignatureAnalysis.h - DXIL semantic signatures -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILSIGNATUREANALYSIS_H
#define LLVM_LIB_TARGET_DIRECTX_DXILSIGNATUREANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Frontend/HLSL/SemanticSignatures.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/VersionTuple.h"

namespace llvm {
class MDTuple;
namespace dxil {

/// Finalized signature information, independent of the source metadata and of
/// instructions replaced by op lowering. Element usage masks are register-
/// relative; dynamic-index masks are element-relative, as in the DXIL ABI.
struct EntrySignature {
  Triple::EnvironmentType Stage;
  SmallVector<hlsl::SemanticSignatureElement> Inputs;
  SmallVector<hlsl::SemanticSignatureElement> Outputs;
  unsigned InputVectors = 0;
  unsigned OutputVectors = 0;
  SmallVector<uint32_t> InputOutputMap;
  bool UseNative16Bit = false;

  MDTuple *getAsMetadata(LLVMContext &Ctx, VersionTuple ValidatorVersion) const;
  SmallVector<uint32_t> getDependencyState() const;
  void print(raw_ostream &OS) const;
};

class ModuleSignatureInfo {
public:
  DenseMap<const Function *, EntrySignature> Entries;
  // The named source metadata is stripped before container emission.
  StringSet<> Names;

  const EntrySignature *get(const Function *F) const {
    auto It = Entries.find(F);
    return It == Entries.end() ? nullptr : &It->second;
  }
  void print(raw_ostream &OS, const Module &M) const;
};

class SignatureAnalysis : public AnalysisInfoMixin<SignatureAnalysis> {
  friend AnalysisInfoMixin<SignatureAnalysis>;
  static AnalysisKey Key;

public:
  using Result = ModuleSignatureInfo;
  Result run(Module &M, ModuleAnalysisManager &AM);
};

class SignatureAnalysisWrapper : public ModulePass {
  std::unique_ptr<ModuleSignatureInfo> Info;

public:
  static char ID;
  SignatureAnalysisWrapper() : ModulePass(ID) {}
  const ModuleSignatureInfo &getSignatureInfo() const { return *Info; }
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override { Info.reset(); }
  void print(raw_ostream &OS, const Module *M) const override;
};

class SignatureAnalysisPrinter
    : public OptionalPassInfoMixin<SignatureAnalysisPrinter> {
  raw_ostream &OS;

public:
  explicit SignatureAnalysisPrinter(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace dxil
} // namespace llvm

#endif
