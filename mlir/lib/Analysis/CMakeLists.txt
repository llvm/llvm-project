set(LLVM_OPTIONAL_SOURCES
  AffineAnalysis.cpp
  AffineStructures.cpp
  CallGraph.cpp
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
  InferTypeOpInterface.cpp
  Liveness.cpp
  SliceAnalysis.cpp
  Dominance.cpp
  Verifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  LINK_LIBS
  MLIRAffineOps
  MLIRIR
  MLIRLoopOps
  MLIRSupport
  LLVMSupport
  )
add_dependencies(MLIRAnalysis
  MLIRAffineOps
  MLIRCallOpInterfacesIncGen
  MLIRTypeInferOpInterfaceIncGen
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

  LINK_LIBS
  MLIRAffineOps
  MLIRIR
  MLIRLoopOps
  MLIRStandardOps
  MLIRSupport
  LLVMSupport
  )
add_dependencies(MLIRLoopAnalysis
  MLIRAffineOps
  MLIRCallOpInterfacesIncGen
  MLIRTypeInferOpInterfaceIncGen
  MLIRLoopOps
  )
