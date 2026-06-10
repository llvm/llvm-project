add_mlir_dialect_library(MLIRTensorTransforms
  BufferizableOpInterfaceImpl.cpp
  ConcatOpPatterns.cpp
  DropRedundantRankExpansionPatterns.cpp
  EmptyOpPatterns.cpp
  ExtractSliceFromReshapeUtils.cpp
  FoldTensorSubsetOps.cpp
  IndependenceTransforms.cpp
  MLGOScalarizeFunctionResult.cpp # MLGO related
  ReshapePatterns.cpp
  RewriteAsConstant.cpp
  RuntimeOpVerification.cpp
  SwapExtractSliceWithProducerPatterns.cpp
  SubsetInsertionOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tensor/Transforms

  DEPENDS
  MLIRTensorTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRArithDialect
  MLIRArithUtils
  MLIRDialectUtils
  MLIRFuncDialect # MLGO related
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRRuntimeVerifiableOpInterface
  MLIRSCFDialect
  MLIRSubsetOpInterface
  MLIRTensorDialect
  MLIRTensorUtils
  MLIRTilingInterface
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorUtils
  MLIRValueBoundsOpInterface
)
