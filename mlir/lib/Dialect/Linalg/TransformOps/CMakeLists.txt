add_mlir_dialect_library(MLIRLinalgTransformOps
  DialectExtension.cpp
  GPUHeuristics.cpp
  LinalgMatchOps.cpp
  LinalgTransformOps.cpp
  Syntax.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg/TransformOps

  DEPENDS
  MLIRLinalgMatchOpsIncGen
  MLIRLinalgTransformOpsIncGen
  MLIRLinalgTransformEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRParser
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  MLIRTransformDialectUtils
  MLIRVectorTransforms
  )
