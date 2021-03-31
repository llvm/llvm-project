add_mlir_conversion_library(MLIRVectorToSCF
  VectorToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToSCF

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRAffineEDSC
  MLIRLLVMIR
  MLIRMemRef
  MLIRTransforms
  )
