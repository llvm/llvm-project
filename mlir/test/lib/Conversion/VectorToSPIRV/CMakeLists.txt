# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestVectorToSPIRV
  TestVectorReductionToSPIRVDotProd.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRVectorToSPIRV
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSPIRVDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
  )
