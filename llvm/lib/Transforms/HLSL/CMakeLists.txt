add_llvm_component_library(LLVMHlsl
  HLSLFinalizeLinkage.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/HLSL

  DEPENDS
  intrinsics_gen
  LLVMAnalysis

  COMPONENT_NAME
  HLSL

  LINK_COMPONENTS
  Analysis
  Core
  Support
  TransformUtils)
