add_mlir_dialect_library(MLIRAffineUtils
  LoopFusionUtils.cpp
  LoopUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRMemRef
  MLIRTransformUtils
  )
