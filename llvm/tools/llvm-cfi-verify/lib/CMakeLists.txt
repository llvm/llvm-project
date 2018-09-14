add_llvm_library(LLVMCFIVerify
  FileAnalysis.cpp
  FileAnalysis.h
  GraphBuilder.cpp
  GraphBuilder.h

  STATIC

  LINK_COMPONENTS
  DebugInfoDWARF
  MC
  MCParser
  Object
  Support
  Symbolize
  )
