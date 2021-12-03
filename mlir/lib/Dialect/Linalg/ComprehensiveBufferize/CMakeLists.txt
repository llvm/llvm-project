set(LLVM_OPTIONAL_SOURCES
  AffineInterfaceImpl.cpp
  ArithInterfaceImpl.cpp
  BufferizableOpInterface.cpp
  BufferizationInterfaceImpl.cpp
  ComprehensiveBufferize.cpp
  LinalgInterfaceImpl.cpp
  ModuleBufferization.cpp
  SCFInterfaceImpl.cpp
  TensorInterfaceImpl.cpp
  VectorInterfaceImpl.cpp
)

add_mlir_dialect_library(MLIRBufferizableOpInterface
  BufferizableOpInterface.cpp

  DEPENDS
  MLIRBufferizableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferization
  MLIRMemRef
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
  MLIRIR
  MLIRLinalg
  MLIRTensor
)

add_mlir_dialect_library(MLIRSCFBufferizableOpInterfaceImpl
  SCFInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRIR
  MLIRSCF
)

add_mlir_dialect_library(MLIRTensorBufferizableOpInterfaceImpl
  TensorInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRIR
  MLIRMemRef
  MLIRTensor
)

add_mlir_dialect_library(MLIRVectorBufferizableOpInterfaceImpl
  VectorInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRIR
  MLIRVector
)

add_mlir_dialect_library(MLIRComprehensiveBufferize
  BufferizationInterfaceImpl.cpp
  ComprehensiveBufferize.cpp
  ModuleBufferization.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemRef
  MLIRStandard
  MLIRStandardOpsTransforms
)
