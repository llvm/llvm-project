add_mlir_library(MLIRPDLLCodeGen
  CPPGen.cpp
  MLIRGen.cpp

  LINK_LIBS PUBLIC
  MLIRParser
  MLIRPDLLAST
  MLIRPDLDialect
  MLIRSupport
  )
