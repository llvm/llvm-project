add_mlir_conversion_library(MLIRMeshToMPI
  MeshToMPI.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MeshToMPI

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRPass
  MLIRMeshDialect
  MLIRMPIDialect
  MLIRTransforms
  )
