add_mlir_conversion_library(MLIRTensorToSPIRV
  TensorToSPIRV.cpp
  TensorToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticToSPIRV
  MLIRFuncToSPIRV
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransformUtils
  MLIRTensor
  )
