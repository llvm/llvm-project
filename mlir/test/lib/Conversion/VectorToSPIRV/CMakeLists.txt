# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestVectorToSPIRV
  TestVectorGatherScatterToSPIRV.cpp
  TestVectorReductionToSPIRVDotProd.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRTestVectorToSPIRV PUBLIC
  MLIRSPIRVConversion
  MLIRUBToSPIRV
  MLIRUBDialect
  MLIRVectorToSPIRV
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSPIRVDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
  )
