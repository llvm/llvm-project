set(LLVM_OPTIONAL_SOURCES
  CanonicalizeGLSLPass.cpp
  DecorateCompositeTypeLayoutPass.cpp
  LowerABIAttributesPass.cpp
  RewriteInsertsPass.cpp
  SPIRVConversion.cpp
  UnifyAliasedResourcePass.cpp
  UpdateVCEPass.cpp
)

add_mlir_dialect_library(MLIRSPIRVConversion
  SPIRVConversion.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV

  LINK_LIBS PUBLIC
  MLIRFunc
  MLIRSPIRV
  MLIRTransformUtils
)

add_mlir_dialect_library(MLIRSPIRVTransforms
  CanonicalizeGLSLPass.cpp
  DecorateCompositeTypeLayoutPass.cpp
  LowerABIAttributesPass.cpp
  RewriteInsertsPass.cpp
  UnifyAliasedResourcePass.cpp
  UpdateVCEPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV

  DEPENDS
  MLIRSPIRVPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRSPIRVConversion
  MLIRSPIRVUtils
)
