add_mlir_dialect_library(MLIRVector
  VectorOps.cpp
  VectorTransferOpTransforms.cpp
  VectorTransforms.cpp
  VectorUtils.cpp
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector

  DEPENDS
  MLIRVectorOpsIncGen
  MLIRVectorOpsEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineEDSC
  MLIREDSC
  MLIRDialectUtils
  MLIRIR
  MLIRStandard
  MLIRAffine
  MLIRLinalg
  MLIRMemRef
  MLIRSCF
  MLIRLoopAnalysis
  MLIRSideEffectInterfaces
  MLIRVectorInterfaces
  )
