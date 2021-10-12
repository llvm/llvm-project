add_mlir_dialect_library(MLIRLinalg
  LinalgInterfaces.cpp
  LinalgOps.cpp
  LinalgTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  DEPENDS
  MLIRLinalgInterfacesIncGen
  MLIRLinalgOpsIncGen
  MLIRLinalgStructuredOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRArithmetic
  MLIRDialectUtils
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRSCF
  MLIRStandard
  MLIRMath
  MLIRMemRef
  MLIRTensor
  MLIRTilingInterface
  MLIRViewLikeInterface
  )
