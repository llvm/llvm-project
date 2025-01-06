add_mlir_dialect_library(MLIRLinalgDialect
  LinalgInterfaces.cpp
  LinalgOps.cpp
  LinalgDialect.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  DEPENDS
  MLIRLinalgInterfacesIncGen
  MLIRLinalgOpsAttributesIncGen
  MLIRLinalgOpsEnumsIncGen
  MLIRLinalgOpsIncGen
  MLIRLinalgStructuredOpsIncGen
  MLIRShardingInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRArithUtils
  MLIRBufferizationDialect
  MLIRDestinationStyleOpInterface
  MLIRDialectUtils
  MLIRFunctionInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRSparseTensorDialect
  MLIRSCFDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
  )
