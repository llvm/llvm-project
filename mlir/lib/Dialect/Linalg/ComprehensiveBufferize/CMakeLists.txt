add_mlir_dialect_library(MLIRModuleBufferization
  ModuleBufferization.cpp

  LINK_LIBS PUBLIC
  MLIRBufferization
  MLIRBufferizationTransforms
  MLIRFunc
  MLIRFuncTransforms
  MLIRIR
  MLIRMemRef
)
