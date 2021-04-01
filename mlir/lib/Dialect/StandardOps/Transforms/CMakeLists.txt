add_mlir_dialect_library(MLIRStandardOpsTransforms
  Bufferize.cpp
  ComposeSubView.cpp
  DecomposeCallGraphTypes.cpp
  ExpandOps.cpp
  FuncBufferize.cpp
  FuncConversions.cpp
  TensorConstantBufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/StandardOps/Transforms

  DEPENDS
  MLIRStandardTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRStandard
  MLIRTransforms
  )
