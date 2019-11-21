add_llvm_component_library(LLVMMIRParser
  MILexer.cpp
  MIParser.cpp
  MIRParser.cpp

  DEPENDS
  intrinsics_gen
  )
