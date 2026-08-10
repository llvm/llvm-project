# Exclude tests from libMLIR.so
add_mlir_library(MLIRTensorTestPasses
  TestTensorTransforms.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRTensorTestPasses PUBLIC
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTransformDialect
  MLIRTransforms
  )
