set(LLVM_OPTIONAL_SOURCES
  AffineAnalysis.cpp
  AffineStructures.cpp
  CallGraph.cpp
  Liveness.cpp
  LoopAnalysis.cpp
  NestedMatcher.cpp
  SliceAnalysis.cpp
  Utils.cpp
  )

add_mlir_library(MLIRAnalysis
  CallGraph.cpp
  Liveness.cpp
  SliceAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  DEPENDS
  mlir-generic-headers
  )

target_link_libraries(MLIRAnalysis
  PUBLIC
  MLIRAffineOps
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRLoopOps
  )

add_mlir_library(MLIRLoopAnalysis
  AffineAnalysis.cpp
  AffineStructures.cpp
  LoopAnalysis.cpp
  NestedMatcher.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  DEPENDS
  mlir-generic-headers
  )

target_link_libraries(MLIRLoopAnalysis
  PUBLIC
  MLIRAffineOps
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRLoopOps)
