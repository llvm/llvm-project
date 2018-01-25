add_llvm_library(LLVMAggressiveInstCombine
  AggressiveInstCombine.cpp
  TruncInstCombine.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/AggressiveInstCombine

  DEPENDS
  intrinsics_gen
  )
