add_mlir_dialect_library(MLIRControlFlowTransforms
  BufferDeallocationOpInterfaceImpl.cpp
  BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ControlFlow/Transforms

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  MLIRMemRefDialect
  MLIRIR
)
