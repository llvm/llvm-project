//===- DirectXTargetMachine.h - DirectX Target Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DIRECTX_H
#define LLVM_LIB_TARGET_DIRECTX_DIRECTX_H

#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {
namespace dxil {
extern cl::opt<bool> PdbInPrivate;
} // namespace dxil

class AttributeMask;
class Function;
class FunctionPass;
class Module;
class ModulePass;
class PassRegistry;
class raw_ostream;

/// Initializer for dxil writer pass
void initializeDXILWriterLegacyPass(PassRegistry &);

/// Initializer for dxil embedder pass
void initializeEmbedDXILPassPass(PassRegistry &);

/// Initializer for DXIL-prepare
void initializeDXILPrepareModuleLegacyPass(PassRegistry &);

/// Pass to convert modules into DXIL-compatable modules
class DXILPrepareModulePass
    : public RequiredPassInfoMixin<DXILPrepareModulePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createDXILPrepareModuleLegacyPass();

/// Pass to downgrade debug information to forms supported by DXIL.
class DXILDebugInfoPass : public OptionalPassInfoMixin<DXILDebugInfoPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXIL debug info lowering.
void initializeDXILDebugInfoLegacyPass(PassRegistry &);

ModulePass *createDXILDebugInfoLegacyPass();

/// Pass to expand intrinsic operations that lack DXIL opCodes
class DXILIntrinsicExpansionPass
    : public OptionalPassInfoMixin<DXILIntrinsicExpansionPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXIL Intrinsic Expansion
void initializeDXILIntrinsicExpansionLegacyPass(PassRegistry &);

ModulePass *createDXILIntrinsicExpansionLegacyPass();

/// Pass to translate loads in the cbuffer address space to cbuffer
/// load intrinsic.
class DXILCBufferAccessPass
    : public OptionalPassInfoMixin<DXILCBufferAccessPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

/// Initializer for DXIL CBuffer Access Pass
void initializeDXILCBufferAccessLegacyPass(PassRegistry &);

ModulePass *createDXILCBufferAccessLegacyPass();

/// Pass to prepare vector data for DXIL by converting vector storage to
/// scalar arrays and lowering dynamically indexed vector accesses to
/// array accesses.
class DXILDataScalarizationPass
    : public OptionalPassInfoMixin<DXILDataScalarizationPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXIL Data Scalarization Pass
void initializeDXILDataScalarizationLegacyPass(PassRegistry &);

ModulePass *createDXILDataScalarizationLegacyPass();

/// Pass to flatten arrays into a one dimensional DXIL legal form
class DXILFlattenArraysPass
    : public OptionalPassInfoMixin<DXILFlattenArraysPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXIL Array Flatten Pass
void initializeDXILFlattenArraysLegacyPass(PassRegistry &);

ModulePass *createDXILFlattenArraysLegacyPass();

/// Pass to eliminate redundant stores and loads from handle globals.
class DXILForwardHandleAccessesPass
    : public OptionalPassInfoMixin<DXILForwardHandleAccessesPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// Initializer for DXIL Forward Handle Accesses Pass
void initializeDXILForwardHandleAccessesLegacyPass(PassRegistry &);

FunctionPass *createDXILForwardHandleAccessesLegacyPass();

/// Pass to rewrite LLVM IR constructs unsupported by DXIL into
/// compatible forms.
class DXILLegalizePass : public OptionalPassInfoMixin<DXILLegalizePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

/// Initializer DXIL legalizationPass
void initializeDXILLegalizeLegacyPass(PassRegistry &);

FunctionPass *createDXILLegalizeLegacyPass();

/// Pass to transform all llvm memory intrinsics to explicit loads and stores.
class DXILMemIntrinsicsPass
    : public OptionalPassInfoMixin<DXILMemIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXIL Mem Intrinsics.
void initializeDXILMemIntrinsicsLegacyPass(PassRegistry &);

ModulePass *createDXILMemIntrinsicsLegacyPass();

/// Pass to lowering LLVM intrinsic call to DXIL op function call.
class DXILOpLoweringPass : public OptionalPassInfoMixin<DXILOpLoweringPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

/// Initializer for DXILOpLowering
void initializeDXILOpLoweringLegacyPass(PassRegistry &);

ModulePass *createDXILOpLoweringLegacyPass();

/// Pass to remove unused DXIL resources and associated
/// global variables.
class DXILRemoveUnusedResourcesPass
    : public OptionalPassInfoMixin<DXILRemoveUnusedResourcesPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// Initializer for DXILRemoveUnusedResources
void initializeDXILRemoveUnusedResourcesLegacyPass(PassRegistry &);

FunctionPass *createDXILRemoveUnusedResourcesLegacyPass();

/// Pass to replace pointers to DXIL resources with load and store
/// operations.
class DXILResourceAccessPass
    : public OptionalPassInfoMixin<DXILResourceAccessPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// Initializer for DXILResourceAccess
void initializeDXILResourceAccessLegacyPass(PassRegistry &);

FunctionPass *createDXILResourceAccessLegacyPass();

/// Pass to assign register slots to resources without binding.
class DXILResourceImplicitBindingPass
    : public OptionalPassInfoMixin<DXILResourceImplicitBindingPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXILResourceImplicitBindingLegacyPass
void initializeDXILResourceImplicitBindingLegacyPass(PassRegistry &);

ModulePass *createDXILResourceImplicitBindingLegacyPass();

/// Pass to emit metadata for DXIL, and to transforms existing LLVM
/// metadata to its DXIL equivalent.
class DXILTranslateMetadataPass
    : public OptionalPassInfoMixin<DXILTranslateMetadataPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXILTranslateMetadata.
void initializeDXILTranslateMetadataLegacyPass(PassRegistry &);

ModulePass *createDXILTranslateMetadataLegacyPass();

/// Pass to pretty print DXIL metadata into IR
/// comments when printing assembly output.
class DXILPrettyPrinterPass
    : public OptionalPassInfoMixin<DXILPrettyPrinterPass> {
  raw_ostream &OS;

public:
  explicit DXILPrettyPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

ModulePass *createDXILPrettyPrinterLegacyPass(raw_ostream &OS);

/// Initializer for DXILPrettyPrinter.
void initializeDXILPrettyPrinterLegacyPass(PassRegistry &);

/// Pass for validating IR after optimizations are applied and before
/// lowering to DXIL.
class DXILPostOptimizationValidationPass
    : public OptionalPassInfoMixin<DXILPostOptimizationValidationPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

/// Initializer for DXILPostOptimizationValidation.
void initializeDXILPostOptimizationValidationLegacyPass(PassRegistry &);

ModulePass *createDXILPostOptimizationValidationLegacyPass();

/// Initializer for dxil::ShaderFlagsAnalysisWrapper pass.
void initializeShaderFlagsAnalysisWrapperPass(PassRegistry &);

/// Initializer for dxil::RootSignatureAnalysisWrapper pass.
void initializeRootSignatureAnalysisWrapperPass(PassRegistry &);

/// Initializer for dxil::SignatureAnalysisWrapper pass.
void initializeSignatureAnalysisWrapperPass(PassRegistry &);

/// Initializer for DXILContainerGlobals pass.
void initializeDXILContainerGlobalsLegacyPass(PassRegistry &);

/// Pass for generating DXContainer part globals.
class DXILContainerGlobalsPass
    : public RequiredPassInfoMixin<DXILContainerGlobalsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createDXILContainerGlobalsLegacyPass();

/// Initializer for DXILContainerPDB pass.
void initializeDXILContainerPDBLegacyPass(PassRegistry &);

/// Pass for emitting DirectX PDB files.
class DXILContainerPDBPass
    : public RequiredPassInfoMixin<DXILContainerPDBPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createDXILContainerPDBLegacyPass();

/// Pass to update the linkage of functions to make sure only
/// shader entry points and exported functions are visible from
/// the module (have program linkage). All other functions will
/// be updated to have internal linkage.
class DXILFinalizeLinkagePass
    : public RequiredPassInfoMixin<DXILFinalizeLinkagePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXILFinalizeLinkage pass.
void initializeDXILFinalizeLinkageLegacyPass(PassRegistry &);

ModulePass *createDXILFinalizeLinkageLegacyPass();
} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DIRECTX_H
