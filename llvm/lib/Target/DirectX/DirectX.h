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
void initializeWriteDXILPassPass(PassRegistry &);

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

/// Pass to downgrade debug information to forms supported by DXIL.
ModulePass *createDXILDebugInfoLegacyPass();

/// Pass to expand intrinsic operations that lack DXIL opCodes
class DXILIntrinsicExpansionPass
    : public OptionalPassInfoMixin<DXILIntrinsicExpansionPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXIL Intrinsic Expansion
void initializeDXILIntrinsicExpansionLegacyPass(PassRegistry &);

/// Pass to expand intrinsic operations that lack DXIL opCodes
ModulePass *createDXILIntrinsicExpansionLegacyPass();

/// Pass to translate loads in the cbuffer address space to intrinsics
class DXILCBufferAccessPass
    : public OptionalPassInfoMixin<DXILCBufferAccessPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

/// Initializer for DXIL CBuffer Access Pass
void initializeDXILCBufferAccessLegacyPass(PassRegistry &);

/// Pass to translate loads in the cbuffer address space to intrinsics
ModulePass *createDXILCBufferAccessLegacyPass();

/// Pass to scalarize llvm global data into a DXIL legal form
class DXILDataScalarizationPass
    : public OptionalPassInfoMixin<DXILDataScalarizationPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXIL Data Scalarization Pass
void initializeDXILDataScalarizationLegacyPass(PassRegistry &);

/// Pass to scalarize llvm global data into a DXIL legal form
ModulePass *createDXILDataScalarizationLegacyPass();

/// Pass to flatten arrays into a one dimensional DXIL legal form
class DXILFlattenArraysPass
    : public OptionalPassInfoMixin<DXILFlattenArraysPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXIL Array Flatten Pass
void initializeDXILFlattenArraysLegacyPass(PassRegistry &);

/// Pass to flatten arrays into a one dimensional DXIL legal form
ModulePass *createDXILFlattenArraysLegacyPass();

/// Pass to eliminate redundant stores and loads from handle globals.
class DXILForwardHandleAccessesPass
    : public OptionalPassInfoMixin<DXILForwardHandleAccessesPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// Initializer for DXIL Forward Handle Accesses Pass
void initializeDXILForwardHandleAccessesLegacyPass(PassRegistry &);

/// Pass to eliminate redundant stores and loads from handle globals.
FunctionPass *createDXILForwardHandleAccessesLegacyPass();

/// Pass to Legalize DXIL by remove i8 truncations and i64 insert/extract
/// elements
class DXILLegalizePass : public OptionalPassInfoMixin<DXILLegalizePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

/// Initializer DXIL legalizationPass
void initializeDXILLegalizeLegacyPass(PassRegistry &);

/// Pass to Legalize DXIL by remove i8 truncations and i64 insert/extract
/// elements
FunctionPass *createDXILLegalizeLegacyPass();

/// Pass to transform all llvm memory intrinsics to explicit loads and stores.
class DXILMemIntrinsicsPass
    : public OptionalPassInfoMixin<DXILMemIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXIL Mem Intrinsics.
void initializeDXILMemIntrinsicsLegacyPass(PassRegistry &);

/// Pass to transform all llvm memory intrinsics to explicit loads and stores.
ModulePass *createDXILMemIntrinsicsLegacyPass();

/// Pass to lowering LLVM intrinsic call to DXIL op function call.
class DXILOpLoweringPass : public OptionalPassInfoMixin<DXILOpLoweringPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

/// Initializer for DXILOpLowering
void initializeDXILOpLoweringLegacyPass(PassRegistry &);

/// Pass to lowering LLVM intrinsic call to DXIL op function call.
ModulePass *createDXILOpLoweringLegacyPass();

/// Pass to update remove unsused resources.
class DXILRemoveUnusedResourcesPass
    : public OptionalPassInfoMixin<DXILRemoveUnusedResourcesPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// Initializer for DXILRemoveUnusedResources
void initializeDXILRemoveUnusedResourcesLegacyPass(PassRegistry &);

/// Pass to update remove unsused resources.
FunctionPass *createDXILRemoveUnusedResourcesLegacyPass();

/// Pass to update resource accesses to use load/store directly.
class DXILResourceAccessPass
    : public OptionalPassInfoMixin<DXILResourceAccessPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// Initializer for DXILResourceAccess
void initializeDXILResourceAccessLegacyPass(PassRegistry &);

/// Pass to update resource accesses to use load/store directly.
FunctionPass *createDXILResourceAccessLegacyPass();

/// Pass to assign register slots to resources without binding.
class DXILResourceImplicitBindingPass
    : public OptionalPassInfoMixin<DXILResourceImplicitBindingPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXILResourceImplicitBindingLegacyPass
void initializeDXILResourceImplicitBindingLegacyPass(PassRegistry &);

/// Pass to assign register slots to resources without binding.
ModulePass *createDXILResourceImplicitBindingLegacyPass();

/// Pass to emit metadata for DXIL.
class DXILTranslateMetadataPass
    : public OptionalPassInfoMixin<DXILTranslateMetadataPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXILTranslateMetadata.
void initializeDXILTranslateMetadataLegacyPass(PassRegistry &);

/// Pass to emit metadata for DXIL.
ModulePass *createDXILTranslateMetadataLegacyPass();

/// Pass to pretty print DXIL metadata.
class DXILPrettyPrinterPass
    : public OptionalPassInfoMixin<DXILPrettyPrinterPass> {
  raw_ostream &OS;

public:
  explicit DXILPrettyPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Pass to pretty print DXIL metadata.
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

/// Pass to lowering LLVM intrinsic call to DXIL op function call.
ModulePass *createDXILPostOptimizationValidationLegacyPass();

/// Initializer for dxil::ShaderFlagsAnalysisWrapper pass.
void initializeShaderFlagsAnalysisWrapperPass(PassRegistry &);

/// Initializer for dxil::RootSignatureAnalysisWrapper pass.
void initializeRootSignatureAnalysisWrapperPass(PassRegistry &);

/// Initializer for dxil::SignatureAnalysisWrapper pass.
void initializeSignatureAnalysisWrapperPass(PassRegistry &);

/// Initializer for DXContainerGlobals pass.
void initializeDXContainerGlobalsPass(PassRegistry &);

/// Pass for generating DXContainer part globals.
ModulePass *createDXContainerGlobalsPass();

/// Initializer for DXContainerPDB pass.
void initializeDXContainerPDBPass(PassRegistry &);

/// Pass for emitting DirectX PDB files.
ModulePass *createDXContainerPDBPass();

/// Pass to finalize linkage of functions.
class DXILFinalizeLinkagePass
    : public RequiredPassInfoMixin<DXILFinalizeLinkagePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Initializer for DXILFinalizeLinkage pass.
void initializeDXILFinalizeLinkageLegacyPass(PassRegistry &);

/// Pass to finalize linkage of functions.
ModulePass *createDXILFinalizeLinkageLegacyPass();
} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DIRECTX_H
