add_llvm_component_library(LLVMCompilerAssistedFuzzing
  FuzzInfo.cpp

  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/CompilerAssistedFuzzing"

  LINK_COMPONENTS
  Support
)
