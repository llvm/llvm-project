set(LLVM_OPTIONAL_SOURCES
  BufferDeallocationOpInterfaceImpl.cpp
  BufferizableOpInterfaceImpl.cpp
  )

add_mlir_dialect_library(MLIRControlFlowTransforms
  StructuralTypeConversions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ControlFlow/Transforms

  LINK_LIBS PUBLIC
  MLIRControlFlowDialect
  MLIRIR
  MLIRTransformUtils
)

add_mlir_dialect_library(MLIRControlFlowBufferizableOpInterfaceImpl
  BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ControlFlow/Transforms

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRIR
)

add_mlir_dialect_library(MLIRControlFlowBufferDeallocationOpInterfaceImpl
  BufferDeallocationOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ControlFlow/Transforms

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMemRefDialect
)
