add_mlir_dialect_library(MLIRMathTransforms
  ExpandTanh.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Math/Transforms

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRStandard
  MLIRTransforms
  )
