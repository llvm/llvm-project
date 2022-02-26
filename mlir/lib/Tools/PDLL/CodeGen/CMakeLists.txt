add_mlir_library(MLIRPDLLCodeGen
  MLIRGen.cpp

  LINK_LIBS PUBLIC
  MLIRParser
  MLIRPDLLAST
  MLIRPDL
  MLIRSupport
  )
