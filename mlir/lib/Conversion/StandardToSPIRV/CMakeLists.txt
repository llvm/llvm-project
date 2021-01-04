add_mlir_conversion_library(MLIRStandardToSPIRV
  LegalizeStandardForSPIRV.cpp
  StandardToSPIRV.cpp
  StandardToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransformUtils
  MLIRStandard
  )
