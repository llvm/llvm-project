add_mlir_dialect_library(MLIRAffineEDSC
  Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIREDSC
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRStandardOps
  )
