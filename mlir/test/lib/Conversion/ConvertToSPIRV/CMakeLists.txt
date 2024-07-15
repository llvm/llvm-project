# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestConvertToSPIRV
  TestSPIRVFuncSignatureConversion.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
  )
