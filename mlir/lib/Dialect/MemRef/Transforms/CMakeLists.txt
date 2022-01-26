add_mlir_dialect_library(MLIRMemRefTransforms
  ComposeSubView.cpp
  ExpandOps.cpp
  FoldSubViewOps.cpp
  NormalizeMemRefs.cpp
  ResolveShapedTypeResultDims.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MemRef

  DEPENDS
  MLIRMemRefPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRArithmetic
  MLIRInferTypeOpInterface
  MLIRMemRef
  MLIRPass
  MLIRStandard
  MLIRTensor
  MLIRTransforms
  MLIRVector
)

