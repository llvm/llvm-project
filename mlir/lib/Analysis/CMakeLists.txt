set(LLVM_OPTIONAL_SOURCES
  AliasAnalysis.cpp
  AffineAnalysis.cpp
  AffineStructures.cpp
  BufferViewFlowAnalysis.cpp
  CallGraph.cpp
  DataFlowAnalysis.cpp
  DataLayoutAnalysis.cpp
  LinearTransform.cpp
  Liveness.cpp
  LoopAnalysis.cpp
  NestedMatcher.cpp
  NumberOfExecutions.cpp
  PresburgerSet.cpp
  SliceAnalysis.cpp
  Utils.cpp

  AliasAnalysis/LocalAliasAnalysis.cpp
  )

add_mlir_library(MLIRAnalysis
  AliasAnalysis.cpp
  BufferViewFlowAnalysis.cpp
  CallGraph.cpp
  DataFlowAnalysis.cpp
  DataLayoutAnalysis.cpp
  Liveness.cpp
  NumberOfExecutions.cpp
  SliceAnalysis.cpp

  AliasAnalysis/LocalAliasAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  )

add_mlir_library(MLIRLoopAnalysis
  AffineAnalysis.cpp
  AffineStructures.cpp
  LinearTransform.cpp
  LoopAnalysis.cpp
  NestedMatcher.cpp
  PresburgerSet.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRPresburger
  MLIRSCF
  )

add_subdirectory(Presburger)
