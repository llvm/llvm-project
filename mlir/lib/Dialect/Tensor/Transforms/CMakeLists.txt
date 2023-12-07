add_mlir_dialect_library(MLIRTensorTransforms
  BufferizableOpInterfaceImpl.cpp
  Bufferize.cpp
  EmptyOpPatterns.cpp
  ExtractSliceFromReshapeUtils.cpp
  FoldIntoPackAndUnpackPatterns.cpp
  FoldTensorSubsetOps.cpp
  IndependenceTransforms.cpp
  MergeConsecutiveInsertExtractSlicePatterns.cpp
  ReshapePatterns.cpp
  RewriteAsConstant.cpp
  SwapExtractSliceWithProducerPatterns.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tensor/Transforms

  DEPENDS
  MLIRTensorTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorUtils
  MLIRTilingInterface
  MLIRTransforms
  MLIRVectorDialect
  MLIRValueBoundsOpInterface
)
