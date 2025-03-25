add_llvm_component_library(LLVMMCLinker
  MCLinkerUtils.cpp
  MCLinker.cpp
  MCPipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ModuleSplitter
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/MCLinker

  LINK_COMPONENTS
  BitReader
  BitWriter
  Core
  IRReader
  Linker
  Support
  TransformUtils
)
