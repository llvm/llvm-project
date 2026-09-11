add_mlir_dialect_library(MLIRVectorHoisting
  VectorHoisting.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector/Transforms

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRValueBoundsOpInterface
  MLIRVectorDialect
  MLIRVectorInterfaces
  MLIRVectorUtils
  )
