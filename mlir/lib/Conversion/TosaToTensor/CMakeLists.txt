add_mlir_conversion_library(MLIRTosaToTensor
  TosaToTensor.cpp
  TosaToTensorPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRTensorDialect
  MLIRTensorUtils
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  MLIRTosaTransforms
  MLIRSupport
  )
