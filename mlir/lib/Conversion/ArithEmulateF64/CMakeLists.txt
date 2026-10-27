add_mlir_conversion_library(MLIRArithEmulateF64
  EmulateF64.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSCFTransforms
  MLIRTransforms
  MLIRVectorDialect
  )
