add_mlir_dialect_library(MLIRBufferizationTransforms
  AllocTensorElimination.cpp
  Bufferize.cpp
  BufferDeallocation.cpp
  BufferOptimizations.cpp
  BufferResultsToOutParams.cpp
  BufferUtils.cpp
  DropEquivalentBufferResults.cpp
  FuncBufferizableOpInterfaceImpl.cpp
  OneShotAnalysis.cpp
  OneShotModuleBufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization

  DEPENDS
  MLIRBufferizationPassIncGen

  LINK_LIBS PUBLIC
  MLIRBufferization
  MLIRControlFlowInterfaces
  MLIRFunc
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemRef
  MLIRPass
  MLIRTensor
  MLIRTransforms
)

