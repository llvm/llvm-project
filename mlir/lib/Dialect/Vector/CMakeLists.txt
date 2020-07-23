add_mlir_dialect_library(MLIRVector
  VectorOps.cpp
  VectorTransforms.cpp
  VectorUtils.cpp
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector

  DEPENDS
  MLIRVectorOpsIncGen
  MLIRVectorTransformPatternsIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  MLIRAffineOps
  MLIRSCF
  MLIRLoopAnalysis
  MLIRSideEffectInterfaces
  MLIRVectorInterfaces
  )
