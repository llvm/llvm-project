set(LLVM_LINK_COMPONENTS
  Support
  TableGen
)

add_mlir_library(MLIRPDLLParser
  Lexer.cpp
  Parser.cpp

  LINK_LIBS PUBLIC
  MLIRPDLLAST
  MLIRSupport
  MLIRTableGen
  )
