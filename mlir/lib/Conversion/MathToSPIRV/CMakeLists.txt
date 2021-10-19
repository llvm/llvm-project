add_mlir_conversion_library(MLIRMathToSPIRV
  MathToSPIRV.cpp
  MathToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMath
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransformUtils
  )

