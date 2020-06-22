add_mlir_conversion_library(MLIRVectorToROCDL
  VectorToROCDL.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToROCDL

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRROCDLIR
  MLIRStandardToLLVM
  MLIRVector
  MLIRTransforms
  )
