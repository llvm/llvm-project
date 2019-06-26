add_llvm_library(LLVMDebugInfoGSYM
  FunctionInfo.cpp
  InlineInfo.cpp
  Range.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/GSYM
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo
  )
