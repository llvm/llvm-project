add_mlir_dialect_library(MLIRBufferizationTransforms
  Bufferize.cpp
  BufferDeallocation.cpp
  BufferOptimizations.cpp
  BufferResultsToOutParams.cpp
  BufferUtils.cpp
  OneShotAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization

  DEPENDS
  MLIRBufferizationPassIncGen

  LINK_LIBS PUBLIC
  MLIRBufferization
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemRef
  MLIRPass
  MLIRStandard
  MLIRTransforms
)
