add_mlir_conversion_library(MLIRStandardToSPIRV
  StandardToSPIRV.cpp
  StandardToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticToSPIRV
  MLIRIR
  MLIRMathToSPIRV
  MLIRMemRef
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransformUtils
  MLIRStandard
  MLIRTensor
  )
