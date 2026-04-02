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
  MLIRGPUDialect
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRMemRefDialect
  MLIRMemRefUtils
  MLIRNVGPUDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRValueBoundsOpInterface
  MLIRVectorDialect
)


# Static type emulation pattern and helper function conflicts.
set_source_files_properties(
  EmulateNarrowType.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
