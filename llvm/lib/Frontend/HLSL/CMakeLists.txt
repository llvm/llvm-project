add_llvm_component_library(LLVMFrontendHLSL
  CBuffer.cpp
  HLSLResource.cpp
  HLSLRootSignature.cpp
  RootSignatureMetadata.cpp
  RootSignatureValidations.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/HLSL

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  BinaryFormat
  Core
  Support
  )
