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

add_llvm_library(MLIRAnalysis
  CallGraph.cpp
  ControlFlowInterfaces.cpp
  InferTypeOpInterface.cpp
  Liveness.cpp
  SliceAnalysis.cpp
  Dominance.cpp
  Verifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  DEPENDS
  MLIRCallOpInterfacesIncGen
  MLIRControlFlowInterfacesIncGen
  MLIRTypeInferOpInterfaceIncGen
  )

target_link_libraries(MLIRAnalysis
  PUBLIC
  MLIRAffineOps
  MLIRLoopOps
  )

add_llvm_library(MLIRLoopAnalysis
  AffineAnalysis.cpp
  AffineStructures.cpp
  LoopAnalysis.cpp
  NestedMatcher.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  DEPENDS
  MLIRCallOpInterfacesIncGen
  MLIRControlFlowInterfacesIncGen
  MLIRTypeInferOpInterfaceIncGen
  )

target_link_libraries(MLIRLoopAnalysis
  PUBLIC
  MLIRAffineOps
  MLIRLoopOps)
