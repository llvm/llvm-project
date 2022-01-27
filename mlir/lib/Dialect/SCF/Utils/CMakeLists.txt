add_mlir_dialect_library(MLIRSCFUtils
  AffineCanonicalizationUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SCF

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRArithmetic
  MLIRDialectUtils
  MLIRIR
  MLIRSCF
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  )
