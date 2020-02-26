add_mlir_dialect_library(MLIRSPIRVTransforms
  DecorateSPIRVCompositeTypeLayoutPass.cpp
  LowerABIAttributesPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  )

target_link_libraries(MLIRSPIRVTransforms
  MLIRPass
  MLIRSPIRV
  )
