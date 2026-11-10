//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def SPIRVOptions : OptionsStruct<prefix = "spirv-">;

// SPIRVAsmPrinter.cpp
defm : EnumField<"spirv-fp-contract", "SPIRVFPContractMode", "SPIRVFPContractMode::On",
  "on,off,fast", ["SPIRVFPContractMode::On", "SPIRVFPContractMode::Off", "SPIRVFPContractMode::Fast"],
  "Override FP contraction policy for SPIR-V kernel entry points">;

// SPIRVAuxDataHandler.cpp
defm : BoolField<"spirv-preserve-auxdata", "false",
  "Preserve LLVM attributes and metadata as NonSemantic.AuxData ExtInst annotations (requires SPV_KHR_non_semantic_info)">;

// SPIRVCtorDtorLowering.cpp
defm : BoolField<"spirv-emit-init-fini-kernel", "true",
  "Emit kernels to call ctor/dtor globals">;
defm : ValueField<"spirv-lower-global-ctor-dtor-id", "StringRef", "",
  "Override unique ID of ctor/dtor globals">;

// SPIRVEmitIntrinsics.cpp
defm : BoolField<"spirv-emit-op-names", "false",
  "Emit OpName for all instructions">;

// SPIRVModuleAnalysis.cpp
defm : EnumListField<"spirv-avoid-capabilities", "SPIRV::Capability::Capability",
  "Shader", ["SPIRV::Capability::Shader"],
  "SPIR-V capabilities to avoid if there are other options enabling a feature">;

// SPIRVNonSemanticDebugHandler.cpp
// When disabled, unwrap a DILexicalBlockFile to its enclosing scope.
defm : BoolField<"spirv-debug-lexical-block-discriminator-scope", "false",
  "Use a DebugLexicalBlockDiscriminator as a DebugScope">;
defm : BoolField<"spirv-debug-scope-forward-refs", "true",
  "Forward-reference debug scopes when SPV_KHR_relaxed_extended_instruction is enabled. May produce invalid SPIR-V if part of a cycle is unsupported; disable to drop the cycle instead">;

// SPIRVRemoveUnusedResources.cpp
defm : BoolField<"spirv-enable-remove-unused-resources", "true",
  "Enable spirv-remove-unused-resources pass">;

// SPIRVSubtarget.cpp
defm : ValueField<"spirv-ext", "StringRef", "",
  "Specify list of enabled SPIR-V extensions">;
defm : BoolField<"spirv-translator-compatibility-mode", "false",
  "SPIR-V Translator compatibility mode">;

// Deprecated no-op kept for backward compatibility. NSDI is emitted when the
// module contains debug info (llvm.dbg.cu).
// TODO: Remove this option after a deprecation period. Callers that used
// -spv-emit-nonsemantic-debug-info should switch to -g.
defm : BoolField<"spv-emit-nonsemantic-debug-info", "false",
  "Deprecated. Use -g to emit SPIR-V NonSemantic.Shader.DebugInfo instructions">;

// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"avoid-spirv-capabilities", "spirv-avoid-capabilities">;
def : AliasOf<"disable-spirv-remove-unused-resources", "spirv-enable-remove-unused-resources=false">;
def : AliasOf<"translator-compatibility-mode", "spirv-translator-compatibility-mode">;
