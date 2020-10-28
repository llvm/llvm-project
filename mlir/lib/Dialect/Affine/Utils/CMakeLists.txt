add_mlir_dialect_library(MLIRAffineUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRTransformUtils
  )
