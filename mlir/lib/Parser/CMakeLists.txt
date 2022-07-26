add_mlir_library(MLIRParser
  Parser.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Parser

  LINK_LIBS PUBLIC
  MLIRAsmParser
  MLIRIR
  )
