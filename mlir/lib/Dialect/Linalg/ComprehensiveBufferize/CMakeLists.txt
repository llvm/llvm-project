set(LLVM_OPTIONAL_SOURCES
  AffineInterfaceImpl.cpp
  ModuleBufferization.cpp
)

add_mlir_dialect_library(MLIRAffineBufferizableOpInterfaceImpl
  AffineInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRBufferization
)

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
