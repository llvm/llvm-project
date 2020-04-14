add_mlir_conversion_library(MLIRStandardToSPIRVTransforms
  ConvertStandardToSPIRV.cpp
  ConvertStandardToSPIRVPass.cpp
  LegalizeStandardForSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen
  )

target_link_libraries(MLIRStandardToSPIRVTransforms
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRSupport
  MLIRTransformUtils
  MLIRSPIRV
  MLIRStandardOps
  )
