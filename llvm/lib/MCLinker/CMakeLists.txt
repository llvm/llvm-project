add_llvm_component_library(LLVMMCLinker
  MCLinker.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ModuleSplitter
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/MCLinker

  LINK_COMPONENTS
  Core
  IRReader
  BitReader
  BitWriter
  Support
  TransformUtils
)
