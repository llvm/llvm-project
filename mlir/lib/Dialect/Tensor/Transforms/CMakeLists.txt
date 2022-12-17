add_mlir_dialect_library(MLIRTensorTransforms
  BufferizableOpInterfaceImpl.cpp
  Bufferize.cpp
  ExtractSliceFromReshapeUtils.cpp
  MergeConsecutiveInsertExtractSlicePatterns.cpp
  ReshapePatterns.cpp
  SplitPaddingPatterns.cpp
  SwapExtractSliceWithProducerPatterns.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tensor/Transforms

  DEPENDS
  MLIRTensorTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
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
  MLIRTilingInterface
  MLIRTransforms
)
