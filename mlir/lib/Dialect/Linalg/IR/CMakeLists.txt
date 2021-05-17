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
  MLIRDialectUtils
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  MLIRStandard
  MLIRMemRef
  MLIRTensor
  )
