add_mlir_conversion_library(MLIRTosaToLinalg
  TosaToLinalg.cpp
  TosaToLinalgPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRDialectUtils
  MLIRIR
  MLIRLinalg
  MLIRLinalgUtils
  MLIRMath
  MLIRPass
  MLIRTensor
  MLIRTosa
  MLIRTosaTransforms
  MLIRSupport
  )
