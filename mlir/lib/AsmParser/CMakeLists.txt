add_mlir_library(MLIRAsmParser
  AffineParser.cpp
  AsmParserState.cpp
  AttributeParser.cpp
  DialectSymbolParser.cpp
  Lexer.cpp
  LocationParser.cpp
  Parser.cpp
  Token.cpp
  TypeParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Parser

  LINK_LIBS PUBLIC
  MLIRIR
  )
