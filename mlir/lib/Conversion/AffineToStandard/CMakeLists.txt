add_mlir_conversion_library(MLIRAffineToStandard
  AffineToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AffineToStandard

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
  )
