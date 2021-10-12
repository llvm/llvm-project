add_mlir_dialect_library(MLIRVector
  VectorOps.cpp
  VectorMultiDimReductionTransforms.cpp
  VectorTransferOpTransforms.cpp
  VectorTransforms.cpp
  VectorUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector

  DEPENDS
  MLIRVectorOpsIncGen
  MLIRVectorOpsEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRDialectUtils
  MLIRIR
  MLIRStandard
  MLIRAffine
  MLIRAffineUtils
  MLIRLinalg
  MLIRMemRef
  MLIRSCF
  MLIRLoopAnalysis
  MLIRDataLayoutInterfaces
  MLIRSideEffectInterfaces
  MLIRVectorInterfaces
  )
