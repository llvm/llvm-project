add_mlir_conversion_library(MLIRUBToSPIRV
  UBToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/UBToSPIRV

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  MLIRUBDialect
  )
