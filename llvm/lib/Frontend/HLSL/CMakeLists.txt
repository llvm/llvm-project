set(LLVM_TARGET_DEFINITIONS
  ${LLVM_MAIN_SRC_DIR}/lib/Target/DirectX/DXIL.td)
llvm_tablegen(DXILIntrinsicStages.inc -gen-dxil-intrinsic-stages)
add_public_tablegen_target(HLSLIntrinsicStagesGen)

add_llvm_component_library(LLVMFrontendHLSL
  CBuffer.cpp
  HLSLBinding.cpp
  HLSLResource.cpp
  HLSLRootSignature.cpp
  IntrinsicStageInfo.cpp
  RootSignatureMetadata.cpp
  RootSignatureValidations.cpp
  SemanticSignaturePacking.cpp
  SemanticSignatures.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/HLSL

  DEPENDS
  HLSLIntrinsicStagesGen
  intrinsics_gen

  LINK_COMPONENTS
  BinaryFormat
  Core
  Support
  )
