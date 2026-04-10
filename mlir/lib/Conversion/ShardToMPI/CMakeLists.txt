add_mlir_conversion_library(MLIRShardToMPI
  ShardToMPI.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ShardToMPI

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgTransforms
  MLIRAffineDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRShardDialect
  MLIRMPIDialect
  MLIRTransforms
  )
