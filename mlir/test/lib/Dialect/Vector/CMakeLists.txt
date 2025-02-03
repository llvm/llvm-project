# Exclude tests from libMLIR.so
add_mlir_library(MLIRVectorTestPasses
  TestVectorTransforms.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRVectorTestPasses PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorToSCF
  MLIRX86VectorDialect
  )
