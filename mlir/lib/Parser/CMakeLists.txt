add_llvm_library(MLIRParser
  Lexer.cpp
  Parser.cpp
  Token.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Parser

  LINK_LIBS
  MLIRIR
  MLIRAnalysis
  LLVMSupport
  )
add_dependencies(MLIRParser MLIRIR MLIRAnalysis)
