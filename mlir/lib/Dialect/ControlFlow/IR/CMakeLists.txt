add_mlir_dialect_library(MLIRControlFlowDialect
  ControlFlowOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ControlFlow/IR

  DEPENDS
  MLIRControlFlowOpsIncGen

  # ControlFlowOps.cpp includes generated Bufferization interfaces and
  # mlir/Transforms/InliningUtils.h.
  HEADER_LIBS
  MLIRBufferizationDialect
  MLIRTransformUtils

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRUBDialect
  )
