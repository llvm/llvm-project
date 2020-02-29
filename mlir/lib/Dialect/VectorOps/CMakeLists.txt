add_mlir_dialect_library(MLIRVectorOps
  VectorOps.cpp
  VectorTransforms.cpp
  VectorUtils.cpp
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/VectorOps

  DEPENDS
  MLIRVectorOpsIncGen
  MLIRVectorTransformPatternsIncGen

  LINK_LIBS
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  MLIRAffineOps
  MLIRLoopOps
  MLIRLoopAnalysis
  LLVMSupport
  )
