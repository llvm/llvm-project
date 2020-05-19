add_mlir_dialect_library(MLIRStandardOpsTransforms
  ExpandAtomic.cpp
  FuncConversions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/StandardOps/Transforms

  DEPENDS
  MLIRStandardTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRStandardOps
  MLIRSupport
  MLIRTransforms
  LLVMSupport
  )
