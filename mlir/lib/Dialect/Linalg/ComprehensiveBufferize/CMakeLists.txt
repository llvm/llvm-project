set(LLVM_OPTIONAL_SOURCES
  AffineInterfaceImpl.cpp
  LinalgInterfaceImpl.cpp
  ModuleBufferization.cpp
  SCFInterfaceImpl.cpp
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

add_mlir_dialect_library(MLIRSCFBufferizableOpInterfaceImpl
  SCFInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferization
  MLIRBufferizationTransforms
  MLIRIR
  MLIRSCF
)

add_mlir_dialect_library(MLIRStdBufferizableOpInterfaceImpl
  StdInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferization
  MLIRStandard
)

add_mlir_dialect_library(MLIRVectorBufferizableOpInterfaceImpl
  VectorInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferization
  MLIRIR
  MLIRVector
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
