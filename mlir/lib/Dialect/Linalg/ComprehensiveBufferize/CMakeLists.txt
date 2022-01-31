set(LLVM_OPTIONAL_SOURCES
  AffineInterfaceImpl.cpp
  LinalgInterfaceImpl.cpp
  ModuleBufferization.cpp
  StdInterfaceImpl.cpp
  VectorInterfaceImpl.cpp
)

add_mlir_dialect_library(MLIRAffineBufferizableOpInterfaceImpl
  AffineInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRBufferization
)

add_mlir_dialect_library(MLIRLinalgBufferizableOpInterfaceImpl
  LinalgInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferization
  MLIRBufferizationTransforms
  MLIRIR
  MLIRLinalg
  MLIRTensor
)

add_mlir_dialect_library(MLIRModuleBufferization
  ModuleBufferization.cpp

  LINK_LIBS PUBLIC
  MLIRBufferization
  MLIRBufferizationTransforms
  MLIRIR
  MLIRMemRef
  MLIRStandard
  MLIRStandardOpsTransforms
)
