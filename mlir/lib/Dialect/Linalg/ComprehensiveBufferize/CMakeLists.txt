set(LLVM_OPTIONAL_SOURCES
  BufferizableOpInterface.cpp
  ComprehensiveBufferize.cpp
  LinalgInterfaceImpl.cpp
)

add_mlir_dialect_library(MLIRBufferizableOpInterface
  BufferizableOpInterface.cpp

  DEPENDS
  MLIRBufferizableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_dialect_library(MLIRLinalgBufferizableOpInterfaceImpl
  LinalgInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRIR
  MLIRLinalg
  MLIRTensor
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
  MLIRTensor
  MLIRVector
)
