add_mlir_conversion_library(MLIRTosaToSCF
  TosaToSCF.cpp
  TosaToSCFPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCF
  MLIRPass
  MLIRTensor
  MLIRTosa
  MLIRTosaTransforms
  MLIRSupport
  )
