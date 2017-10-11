add_llvm_library(LLVMCFIVerify
  FileAnalysis.cpp
  FileAnalysis.h

  LINK_COMPONENTS
  MC
  MCParser
  Object
  Support)
