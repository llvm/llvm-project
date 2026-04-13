# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestVectorToSPIRV
  TestVectorReductionToSPIRVDotProd.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRTestVectorToSPIRV PUBLIC
  MLIRVectorToSPIRV
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSPIRVDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
  )
