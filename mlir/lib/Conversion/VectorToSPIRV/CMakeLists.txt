add_mlir_conversion_library(MLIRVectorToSPIRV
  VectorToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToSPIRV

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_LIBS PUBLIC
  MLIRSPIRV
  MLIRVector
  MLIRTransforms
  )
