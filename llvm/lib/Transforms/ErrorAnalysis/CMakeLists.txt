add_subdirectory(AtomicCondition)

add_llvm_component_library(LLVMErrorAnalysis
  ErrorAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/ErrorAnalysis

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  Support
  AtomicCondition
  )

