add_mlir_dialect_library(MLIRVectorUtils
  VectorUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector/Utils

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRArithmeticDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRVectorInterfaces
  )
