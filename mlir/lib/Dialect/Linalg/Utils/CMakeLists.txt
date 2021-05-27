add_mlir_dialect_library(MLIRLinalgUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRIR
  MLIRLinalg
  MLIRSCF
  MLIRPass
  MLIRStandard
  MLIRTransformUtils
  )
