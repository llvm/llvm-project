set(LLVM_OPTIONAL_SOURCES
  BufferizableOpInterface.cpp
  ComprehensiveBufferize.cpp
  LinalgInterfaceImpl.cpp
  TensorInterfaceImpl.cpp
  VectorInterfaceImpl.cpp
)

add_mlir_dialect_library(MLIRBufferizableOpInterface
  BufferizableOpInterface.cpp

  DEPENDS
  MLIRBufferizableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRef
)

add_mlir_dialect_library(MLIRLinalgBufferizableOpInterfaceImpl
  LinalgInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRIR
  MLIRLinalg
  MLIRTensor
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
  ComprehensiveBufferize.cpp

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRBufferizableOpInterface
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemRef
  MLIRSCF
  MLIRStandard
  MLIRStandardOpsTransforms
)
