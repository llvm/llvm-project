add_mlir_dialect_library(MLIRLinalgUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineAnalysis
  MLIRAffineUtils
  MLIRArithmetic
  MLIRIR
  MLIRLinalg
  MLIRSCF
  MLIRPass
  MLIRTensorUtils
  MLIRTransformUtils
  )
