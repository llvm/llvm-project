add_mlir_dialect_library(MLIRMemRefTransforms
  AllocationOpInterfaceImpl.cpp
  BufferViewFlowOpInterfaceImpl.cpp
  ComposeSubView.cpp
  ExpandOps.cpp
  ExpandRealloc.cpp
  ExpandStridedMetadata.cpp
  EmulateWideInt.cpp
  EmulateNarrowType.cpp
  ExtractAddressComputations.cpp
  FoldMemRefAliasOps.cpp
  IndependenceTransforms.cpp
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
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRArithDialect
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRMemRefDialect
  MLIRNVGPUDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRValueBoundsOpInterface
  MLIRVectorDialect
)

