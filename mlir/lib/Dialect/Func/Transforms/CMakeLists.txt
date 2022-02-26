add_mlir_dialect_library(MLIRFuncTransforms
  DecomposeCallGraphTypes.cpp
  FuncBufferize.cpp
  FuncConversions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Func/Transforms

  DEPENDS
  MLIRFuncTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRArithmeticTransforms
  MLIRBufferization
  MLIRBufferizationTransforms
  MLIRFunc
  MLIRIR
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRTransforms
  )
