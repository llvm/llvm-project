add_mlir_dialect_library(MLIRAGXTransforms
  ExpandOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AGX/Transforms

  DEPENDS
  MLIRAGXTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAGXDialect
  MLIRAGXUtils
  MLIRAGXIRDialect
  MLIRArithDialect
  MLIRGPUDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRVectorDialect
  MLIRTransformUtils
  )
