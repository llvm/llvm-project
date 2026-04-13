add_mlir_dialect_library(MLIRAffineDialect
  AffineMemoryOpInterfaces.cpp
  AffineOps.cpp
  AffineValueMap.cpp
  InferIntRangeInterfaceImpls.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineMemoryOpInterfacesIncGen
  MLIRAffineOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRMemRefDialect
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRUBDialect
  MLIRValueBoundsOpInterface
  )
