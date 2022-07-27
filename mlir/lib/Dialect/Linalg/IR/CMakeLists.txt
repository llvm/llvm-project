add_mlir_dialect_library(MLIRLinalgDialect
  LinalgInterfaces.cpp
  LinalgOps.cpp
  LinalgDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  DEPENDS
  MLIRLinalgInterfacesIncGen
  MLIRLinalgOpsAttributesIncGen
  MLIRLinalgOpsEnumsIncGen
  MLIRLinalgOpsIncGen
  MLIRLinalgStructuredOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithmeticDialect
  MLIRArithmeticUtils
  MLIRBufferizationDialect
  MLIRDialectUtils
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRSparseTensorDialect
  MLIRSCFDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRTilingInterface
  MLIRViewLikeInterface
  )
