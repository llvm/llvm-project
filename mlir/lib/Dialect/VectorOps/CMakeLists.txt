add_mlir_dialect_library(MLIRVectorOps
  VectorOps.cpp
  VectorTransforms.cpp
  VectorUtils.cpp
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/VectorOps

  LINK_LIBS
  MLIRIR
  MLIRStandardOps
  MLIRAffineOps
  MLIRLoopOps
  MLIRLoopAnalysis
  LLVMSupport
  )

add_dependencies(MLIRVectorOps MLIRVectorOpsIncGen)
add_dependencies(MLIRVectorOps MLIRVectorTransformPatternsIncGen)
