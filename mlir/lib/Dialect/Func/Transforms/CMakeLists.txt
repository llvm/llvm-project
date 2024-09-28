add_mlir_dialect_library(MLIRFuncTransforms
  DecomposeCallGraphTypes.cpp
  DuplicateFunctionElimination.cpp
  FuncBufferize.cpp
  FuncConversions.cpp
  OneToNFuncConversions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Func/Transforms

  DEPENDS
  MLIRFuncTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
  )
