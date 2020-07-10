add_mlir_dialect_library(MLIRSPIRVTransforms
  DecorateSPIRVCompositeTypeLayoutPass.cpp
  LowerABIAttributesPass.cpp
  RewriteInsertsPass.cpp
  UpdateVCEPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV

  DEPENDS
  MLIRSPIRVPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRSPIRV
  )
