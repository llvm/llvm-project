add_mlir_dialect_library(MLIRMemRefDialect
  MemRefDialect.cpp
  MemRefOps.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/inlude/mlir/Dialect/MemRefDialect

  DEPENDS
  MLIRMemRefOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithUtils
  MLIRControlFlowInterfaces
  MLIRDialect
  MLIRDialectUtils
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
)
