add_mlir_dialect_library(MLIRBufferization
  PARTIAL_SOURCES_INTENDED
  AllocationOpInterface.cpp
  BufferizationOps.cpp
  BufferizationDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization

  DEPENDS
  MLIRAllocationOpInterfaceIncGen
  MLIRBufferizationOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRTensor
  MLIRMemRef
  )

add_mlir_dialect_library(MLIRBufferizableOpInterface
  PARTIAL_SOURCES_INTENDED
  BufferizableOpInterface.cpp
  BufferizationInterfaceImpl.cpp

  DEPENDS
  MLIRBufferizableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferization
  MLIRMemRef
)
