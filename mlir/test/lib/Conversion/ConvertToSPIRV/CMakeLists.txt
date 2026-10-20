# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestConvertToSPIRV
  TestSPIRVFuncSignatureConversion.cpp
  TestSPIRVVectorUnrolling.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRTestConvertToSPIRV PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRPass
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  MLIRTransformUtils
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorTransforms
  )
