add_mlir_conversion_library(MLIRIndexToSPIRV
  IndexToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/IndexToSPIRV

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIndexDialect
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  )
