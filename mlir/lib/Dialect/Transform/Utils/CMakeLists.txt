add_mlir_dialect_library(MLIRTransformDialectUtils
  DiagnosedSilenceableFailure.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Transform

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRSupport
  MLIRViewLikeInterface
)
