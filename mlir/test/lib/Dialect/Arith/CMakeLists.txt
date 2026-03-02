# Exclude tests from libMLIR.so
add_mlir_library(MLIRArithTestPasses
  TestEmulateWideInt.cpp

  EXCLUDE_FROM_LIBMLIR
)
mlir_target_link_libraries(MLIRArithTestPasses PUBLIC
  MLIRArithDialect
  MLIRArithTransforms
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRVectorDialect
)
