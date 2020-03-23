add_mlir_dialect_library(MLIRAffine
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineOpsIncGen
  )
target_link_libraries(MLIRAffine
  PUBLIC
  MLIREDSC
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffects
  MLIRStandardOps
  )
