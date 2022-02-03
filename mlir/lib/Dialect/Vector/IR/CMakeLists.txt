add_mlir_dialect_library(MLIRVector
  VectorOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector/IR

  DEPENDS
  MLIRVectorOpsIncGen
  MLIRVectorOpsEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineAnalysis
  MLIRAffineUtils
  MLIRArithmetic
  MLIRDialectUtils
  MLIRIR
  MLIRMemRef
  MLIRStandard
  MLIRSideEffectInterfaces
  MLIRVectorInterfaces
  )
