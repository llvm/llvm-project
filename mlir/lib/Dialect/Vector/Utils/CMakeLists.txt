add_mlir_dialect_library(MLIRVectorUtils
  VectorUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector/Utils

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineAnalysis
  MLIRArithmetic
  MLIRDialectUtils
  MLIRIR
  MLIRMemRef
  MLIRStandard
  MLIRTensor
  MLIRVector
  MLIRVectorInterfaces
  )
