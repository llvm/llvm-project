add_llvm_library(emb_utils STATIC
  utils.cpp
  
  LINK_COMPONENTS
  Analysis
  Core
  Support
  IRReader
  AsmParser
  Passes
  Demangle
)
