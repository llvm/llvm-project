add_mlir_conversion_library(MLIRStandardToStandard
  StandardToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToStandard
  )
target_link_libraries(MLIRStandardToStandard
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRStandardOps
  MLIRTransforms
  )
