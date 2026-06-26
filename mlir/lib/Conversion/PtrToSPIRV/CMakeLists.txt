add_mlir_conversion_library(MLIRPtrToSPIRV
  PtrToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/PtrToSPIRV

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRPtrDialect
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  )
