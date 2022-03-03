add_mlir_conversion_library(MLIRArithmeticToSPIRV
  ArithmeticToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithmeticToSPIRV

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRFuncToSPIRV
  MLIRSPIRVConversion
  MLIRSPIRV
  )
