set(LLVM_OPTIONAL_SOURCES
  AffineAnalysis.cpp
  AffineStructures.cpp
  CallGraph.cpp
  Dominance.cpp
  Liveness.cpp
  LoopAnalysis.cpp
  NestedMatcher.cpp
  SliceAnalysis.cpp
  Utils.cpp
  Verifier.cpp
  )

add_mlir_library(MLIRAnalysis
  CallGraph.cpp
  Liveness.cpp
  SliceAnalysis.cpp
  Dominance.cpp
  Verifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis
  )

target_link_libraries(MLIRAnalysis
  PUBLIC
  MLIRAffine
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
  )

target_link_libraries(MLIRLoopAnalysis
  PUBLIC
  MLIRAffine
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRLoopOps)
