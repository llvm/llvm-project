add_mlir_dialect_library(MLIRBufferizationTransforms
  Bufferize.cpp
  BufferDeallocation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization

  DEPENDS
  MLIRBufferizationPassIncGen

  LINK_LIBS PUBLIC
  MLIRBufferization
  MLIRPass
  MLIRTransforms
)
