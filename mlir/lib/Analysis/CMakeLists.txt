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
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRPresburger
  MLIRSCF
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
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRPresburger
  MLIRSCF
  )
  
add_subdirectory(Presburger)
