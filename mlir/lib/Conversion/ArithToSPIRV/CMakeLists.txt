add_mlir_conversion_library(MLIRArithToSPIRV
  ArithToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToSPIRV

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncToSPIRV
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  )
