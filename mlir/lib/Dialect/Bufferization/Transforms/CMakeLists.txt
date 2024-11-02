add_mlir_dialect_library(MLIRBufferizationTransforms
  AllocTensorElimination.cpp
  Bufferize.cpp
  BufferDeallocation.cpp
  BufferOptimizations.cpp
  BufferResultsToOutParams.cpp
  BufferUtils.cpp
  BufferViewFlowAnalysis.cpp
  DropEquivalentBufferResults.cpp
  EmptyTensorToAllocTensor.cpp
  FuncBufferizableOpInterfaceImpl.cpp
  OneShotAnalysis.cpp
  OneShotModuleBufferize.cpp
  TensorCopyInsertion.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization

  DEPENDS
  MLIRBufferizationPassIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRViewLikeInterface
)

