add_mlir_dialect_library(MLIRAffineOps
  AffineOps.cpp
  AffineValueMap.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineOpsIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRStandardOps
  )
