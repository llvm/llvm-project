set(LLVM_OPTIONAL_SOURCES
  AffineAnalysis.cpp
  AffineStructures.cpp
  CallGraph.cpp
  ControlFlowInterfaces.cpp
  Dominance.cpp
  InferTypeOpInterface.cpp
  Liveness.cpp
  LoopAnalysis.cpp
  NestedMatcher.cpp
  SliceAnalysis.cpp
  Utils.cpp
  Verifier.cpp
  )

add_mlir_library(MLIRControlFlowAnalysis
  ControlFlowInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  DEPENDS
  MLIRControlFlowInterfacesIncGen
  )
target_link_libraries(MLIRControlFlowAnalysis
  PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRAnalysis
  CallGraph.cpp
  InferTypeOpInterface.cpp
  Liveness.cpp
  SliceAnalysis.cpp
  Dominance.cpp
  Verifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  DEPENDS
  MLIRCallOpInterfacesIncGen
  MLIRTypeInferOpInterfaceIncGen
  )

target_link_libraries(MLIRAnalysis
  PUBLIC
  MLIRAffineOps
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
  MLIRCallOpInterfacesIncGen
  MLIRTypeInferOpInterfaceIncGen
  )

target_link_libraries(MLIRLoopAnalysis
  PUBLIC
  MLIRAffineOps
  MLIRLoopOps)
