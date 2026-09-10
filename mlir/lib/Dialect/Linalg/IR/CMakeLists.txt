add_mlir_dialect_library(MLIRLinalgDialect
  LinalgInterfaces.cpp
  LinalgOps.cpp
  LinalgDialect.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  PRECOMPILE_HEADERS
  [["mlir/Dialect/Linalg/IR/pch.h"]]

  DEPENDS
  MLIRLinalgInterfacesIncGen
  MLIRLinalgOpsAttributesIncGen
  MLIRLinalgOpsEnumsIncGen
  MLIRLinalgOpsIncGen
  MLIRLinalgStructuredOpsIncGen
  MLIRLinalgRelayoutOpsIncGen
  MLIRShardingInterfaceIncGen
  MLIRRelayoutOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRArithUtils
  MLIRBufferizationDialect
  MLIRDestinationStyleOpInterface
  MLIRDialectUtils
  MLIRFunctionInterfaces
  MLIRIndexingMapOpInterface
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
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
  )
