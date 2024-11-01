add_mlir_dialect_library(MLIRMemRefTransforms
  BufferizableOpInterfaceImpl.cpp
  ComposeSubView.cpp
  ExpandOps.cpp
  ExpandStridedMetadata.cpp
  EmulateWideInt.cpp
  FoldMemRefAliasOps.cpp
  MultiBuffer.cpp
  NormalizeMemRefs.cpp
  ResolveShapedTypeResultDims.cpp
  RuntimeOpVerification.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MemRef

  DEPENDS
  MLIRMemRefPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRMemRefDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRVectorDialect
)

