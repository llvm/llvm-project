add_mlir_conversion_library(MLIRVectorToLoops
  ConvertVectorToLoops.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToLoops

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRAffineEDSC
  MLIRLLVMIR
  MLIRTransforms
  )
