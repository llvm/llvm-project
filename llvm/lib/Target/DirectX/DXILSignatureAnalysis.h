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
namespace mcdxbc {
class Signature;
struct PSVRuntimeInfo;
} // namespace mcdxbc

namespace dxil {

struct EntrySignature {
  Triple::EnvironmentType Stage;
  SmallVector<hlsl::SemanticSignatureElement> Inputs;
  SmallVector<hlsl::SemanticSignatureElement> Outputs;
  unsigned InputVectors = 0;
  unsigned OutputVectors = 0;
  SmallVector<uint32_t> InputOutputMap;
  bool UseNative16Bit = false;

  /// Return the DXIL input/output/patch-constant signature triple. Empty input
  /// and output lists become null operands; return null if both lists are
  /// empty. ValidatorVersion controls which access-mask properties are emitted.
  MDTuple *getAsMetadata(LLVMContext &Ctx, VersionTuple ValidatorVersion) const;

  /// Append ISG1/OSG1 parameters to the supplied builders. ValidatorVersion
  /// selects legacy component-type and exclusive-mask encoding conventions.
  void buildSignatures(mcdxbc::Signature &Input, mcdxbc::Signature &Output,
                       VersionTuple ValidatorVersion) const;

  /// Append PSV signature elements and set vector extents, the dependency map,
  /// and stage-specific signature flags. Does not finalize or serialize PSV.
  void updatePSV(mcdxbc::PSVRuntimeInfo &PSV,
                 VersionTuple ValidatorVersion) const;

  /// Return the serialized dx.viewIdState words for this entry: input scalar
  /// count, output scalar count, then the input/output dependency table. Scalar
  /// counts include padding and equal four times the respective vector extent.
  /// Each input scalar has ceil(output scalar count / 32) consecutive words,
  /// where bit (4 * row + column) identifies an output scalar.
  SmallVector<uint32_t> getDependencyState() const;
  void print(raw_ostream &OS) const;
};

class ModuleSignatureInfo {
public:
  ModuleSignatureInfo() = default;
  ModuleSignatureInfo(const ModuleSignatureInfo &) = delete;
  ModuleSignatureInfo &operator=(const ModuleSignatureInfo &) = delete;
  ModuleSignatureInfo(ModuleSignatureInfo &&) = default;
  ModuleSignatureInfo &operator=(ModuleSignatureInfo &&) = default;

  DenseMap<const Function *, EntrySignature> Entries;
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
