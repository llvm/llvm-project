add_llvm_component_library(LLVMModuleSplitter
  ModuleSplitter.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ModuleSplitter

  LINK_COMPONENTS
  Core
  IRReader
  LLVMBitReader
  LLVMBitWriter
  LLVMModuleSplitter
  Support
  TransformUtils
)
