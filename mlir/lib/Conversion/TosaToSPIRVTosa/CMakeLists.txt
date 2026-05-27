add_mlir_conversion_library(MLIRTosaToSPIRVTosa
  TosaToSPIRVTosa.cpp
  TosaToSPIRVTosaOps.cpp
  TosaToSPIRVTosaPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransformUtils
  MLIRTosaDialect
)
