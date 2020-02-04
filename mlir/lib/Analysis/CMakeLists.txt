add_llvm_library(MLIRAnalysis STATIC
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

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis
  )
add_dependencies(MLIRAnalysis
  MLIRAffineOps
  MLIRCallOpInterfacesIncGen
  MLIRTypeInferOpInterfaceIncGen
  MLIRLoopOps
  )
target_link_libraries(MLIRAnalysis MLIRAffineOps MLIRLoopOps)
