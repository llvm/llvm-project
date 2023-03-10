# Exclude tests from libMLIR.so
add_mlir_library(MLIRVectorTestPasses
  TestVectorTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUOps
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorToSCF
  MLIRX86VectorDialect
  )
