set(LLVM_OPTIONAL_SOURCES
  BufferizableOpInterface.cpp
  ComprehensiveBufferize.cpp
)

add_mlir_dialect_library(MLIRBufferizableOpInterface
  BufferizableOpInterface.cpp

  DEPENDS
  MLIRBufferizableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_dialect_library(MLIRComprehensiveBufferize
  ComprehensiveBufferize.cpp

  LINK_LIBS PUBLIC
  MLIRBufferizableOpInterface
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemRef
  MLIRLinalg
  MLIRSCF
  MLIRStandard
  MLIRStandardOpsTransforms
  MLIRTensor
  MLIRVector
)
