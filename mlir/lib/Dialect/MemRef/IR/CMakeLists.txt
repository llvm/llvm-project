add_mlir_dialect_library(MLIRMemRefDialect
  MemRefAttrs.cpp
  MemRefDialect.cpp
  MemRefMemorySlot.cpp
  MemRefOps.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/inlude/mlir/Dialect/MemRefDialect

  DEPENDS
  MLIRMemRefAttrsIncGen
  MLIRMemRefInterfacesIncGen
  MLIRMemRefOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithUtils
  MLIRControlFlowInterfaces
  MLIRDialect
  MLIRDialectUtils
  MLIRInferIntRangeCommon
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemorySlotInterfaces
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
)
