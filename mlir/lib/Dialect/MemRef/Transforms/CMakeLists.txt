add_mlir_dialect_library(MLIRMemRefTransforms
  AllocationOpInterfaceImpl.cpp
  BufferViewFlowOpInterfaceImpl.cpp
  ComposeSubView.cpp
  ElideReinterpretCast.cpp
  ExpandOps.cpp
  ExpandRealloc.cpp
  ExpandStridedMetadata.cpp
  EmulateWideInt.cpp
  EmulateNarrowType.cpp
  ExtractAddressComputations.cpp
  FlattenMemRefs.cpp
  FoldMemRefAliasOps.cpp
  IndependenceTransforms.cpp
  MultiBuffer.cpp
  NormalizeMemRefs.cpp
  ReifyResultShapes.cpp
  ResolveShapedTypeResultDims.cpp
  RuntimeOpVerification.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MemRef

  DEPENDS
  MLIRMemRefPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRArithDialect
  MLIRArithTransforms
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRMemRefDialect
  MLIRMemRefUtils
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRValueBoundsOpInterface
  MLIRVectorDialect
)
