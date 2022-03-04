add_mlir_library(MLIRPDLLParser
  Lexer.cpp
  Parser.cpp
  
  LINK_COMPONENTS
  Support
  TableGen

  LINK_LIBS PUBLIC
  MLIRPDLLAST
  MLIRSupport
  MLIRTableGen
  )
