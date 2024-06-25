add_mlir_dialect_library(MLIRTensorTransforms
  BufferizableOpInterfaceImpl.cpp
  ConcatOpPatterns.cpp
  EmptyOpPatterns.cpp
  ExtractSliceFromReshapeUtils.cpp
  FoldTensorSubsetOps.cpp
  IndependenceTransforms.cpp
  MergeConsecutiveInsertExtractSlicePatterns.cpp
  PackAndUnpackPatterns.cpp
  ReshapePatterns.cpp
  RewriteAsConstant.cpp
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
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRDialectUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
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
