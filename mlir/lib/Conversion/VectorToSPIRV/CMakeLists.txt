add_mlir_conversion_library(MLIRVectorToSPIRV
  VectorToSPIRV.cpp
  VectorToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToSPIRV

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_LIBS PUBLIC
  MLIRSPIRV
  MLIRSPIRVConversion
  MLIRVector
  MLIRTransforms
  )
