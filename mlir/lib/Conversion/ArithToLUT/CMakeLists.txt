add_mlir_conversion_library(MLIRArithToLUT
  ArithToLUT.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToLUT

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
  MLIRVectorDialect
)
