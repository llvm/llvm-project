add_mlir_dialect_library(MLIRStandardOpsTransforms
  Bufferize.cpp
  DecomposeCallGraphTypes.cpp
  FuncBufferize.cpp
  FuncConversions.cpp
  TensorConstantBufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/StandardOps/Transforms

  DEPENDS
  MLIRStandardTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRArithmeticTransforms
  MLIRBufferizationTransforms
  MLIRIR
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRStandard
  MLIRTransforms
  )
