add_llvm_component_library(LLVMCASUtil
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CASObjectFormats

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Support
  CAS
  BinaryFormat
)
