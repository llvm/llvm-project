add_mlir_conversion_library(MLIRComplexDivisionConversion
  DivisionConverter.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRComplexDialect
  MLIRLLVMDialect
  MLIRMathDialect
  )
