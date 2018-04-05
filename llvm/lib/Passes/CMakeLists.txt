add_llvm_library(LLVMPasses
  PassBuilder.cpp
  PassPlugin.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Passes

  DEPENDS
  intrinsics_gen
  )
