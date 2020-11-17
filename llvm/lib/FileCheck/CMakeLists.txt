add_llvm_component_library(LLVMFileCheck
  FileCheck.cpp

  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/FileCheck"

  LINK_COMPONENTS
  Support
)
