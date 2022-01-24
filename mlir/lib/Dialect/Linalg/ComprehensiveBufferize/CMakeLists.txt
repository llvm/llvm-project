set(LLVM_OPTIONAL_SOURCES
  AffineInterfaceImpl.cpp
  ArithInterfaceImpl.cpp
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
  MLIRBufferizableOpInterface
)

add_mlir_dialect_library(MLIRArithBufferizableOpInterfaceImpl
  ArithInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRBufferizableOpInterface
  MLIRIR
  MLIRMemRef
  MLIRStandardOpsTransforms
)

add_mlir_dialect_library(MLIRLinalgBufferizableOpInterfaceImpl
  LinalgInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRBufferizationTransforms
  MLIRIR
  MLIRLinalg
  MLIRTensor
)

add_mlir_dialect_library(MLIRSCFBufferizableOpInterfaceImpl
  SCFInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRBufferizationTransforms
  MLIRIR
  MLIRSCF
)

add_mlir_dialect_library(MLIRStdBufferizableOpInterfaceImpl
  StdInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRStandard
)

add_mlir_dialect_library(MLIRVectorBufferizableOpInterfaceImpl
  VectorInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRIR
  MLIRVector
)

add_mlir_dialect_library(MLIRModuleBufferization
  ModuleBufferization.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRBufferizationTransforms
  MLIRIR
  MLIRMemRef
  MLIRStandard
  MLIRStandardOpsTransforms
)
