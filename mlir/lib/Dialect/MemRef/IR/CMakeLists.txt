add_mlir_dialect_library(MLIRMemRefDialect
  MemRefDialect.cpp
  MemRefMemorySlot.cpp
  MemRefOps.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/inlude/mlir/Dialect/MemRef/IR

  DEPENDS
  MLIRMemRefOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithUtils
  MLIRControlFlowInterfaces
  MLIRDialect
  MLIRDialectUtils
  MLIRInferIntRangeCommon
  MLIRInferIntRangeInterface
  MLIRInferStridedMetadataInterface
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemOpInterfaces
  MLIRMemorySlotInterfaces
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
)
