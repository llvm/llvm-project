add_mlir_conversion_library(MLIRLinalgToSPIRV
  LinalgToSPIRV.cpp
  LinalgToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalg
  MLIRLinalgUtils
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVConversion
  MLIRSupport
  )
