add_mlir_dialect_library(MLIRLinalgOps
  LinalgOps.cpp
  LinalgTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  DEPENDS
  MLIRLinalgOpsIncGen
  MLIRLinalgStructuredOpsIncGen
  MLIRLinalgStructuredOpsInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  MLIRStandardOps
  )
