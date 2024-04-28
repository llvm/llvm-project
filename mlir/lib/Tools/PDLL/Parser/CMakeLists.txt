
llvm_add_library(MLIRPDLLParser STATIC
  CodeComplete.cpp
  Lexer.cpp
  Parser.cpp
  
  DISABLE_LLVM_LINK_LLVM_DYLIB

  LINK_COMPONENTS
  Support
  TableGen

  LINK_LIBS PUBLIC
  MLIRPDLLAST
  MLIRSupport
  MLIRTableGen
  )

mlir_check_all_link_libraries(MLIRPDLLParser)

add_mlir_library_install(MLIRPDLLParser)
