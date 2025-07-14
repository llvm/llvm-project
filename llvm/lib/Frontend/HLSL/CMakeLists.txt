add_llvm_component_library(LLVMFrontendHLSL
  CBuffer.cpp
  HLSLResource.cpp
  HLSLRootSignatureUtils.cpp

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
