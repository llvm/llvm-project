add_mlir_conversion_library(MLIRTosaToStandard
  TosaToStandard.cpp
  TosaToStandardPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRIR
  MLIRPass
  MLIRTosa
  MLIRTosaTransforms
  MLIRSupport
  )
