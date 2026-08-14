add_mlir_conversion_library(MLIRSCFToAffine
  SCFToAffine.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToAffine

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTransforms
  )
