add_mlir_library(MLIRParser
  Lexer.cpp
  Parser.cpp
  Token.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Parser
  )
target_link_libraries(MLIRParser
  PUBLIC
  MLIRIR
  MLIRAnalysis)
