set(LLVM_OPTIONAL_SOURCES
  AllocationOpInterfaceImpl.cpp
  BufferViewFlowOpInterfaceImpl.cpp
  )

add_mlir_dialect_library(MLIRMemRefTransforms
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
  MemorySlotInterfacesImpl.cpp
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
  MLIRMemorySlotInterfaces
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRValueBoundsOpInterface
  MLIRVectorDialect
)

add_mlir_dialect_library(MLIRMemRefAllocationOpInterfaceImpl
  AllocationOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MemRef

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRIR
  MLIRMemRefDialect
)

add_mlir_dialect_library(MLIRMemRefBufferViewFlowOpInterfaceImpl
  BufferViewFlowOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MemRef

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRIR
  MLIRMemRefDialect
)
