set(LLVM_OPTIONAL_SOURCES
  AliasAnalysis.cpp
  CallGraph.cpp
  DataLayoutAnalysis.cpp
  FlatLinearValueConstraints.cpp
  Liveness.cpp
  CFGLoopInfo.cpp
  SliceAnalysis.cpp

  AliasAnalysis/LocalAliasAnalysis.cpp

  DataFlow/ConstantPropagationAnalysis.cpp
  DataFlow/DeadCodeAnalysis.cpp
  DataFlow/DenseAnalysis.cpp
  DataFlow/IntegerRangeAnalysis.cpp
  DataFlow/SparseAnalysis.cpp
  )

add_subdirectory(Presburger)

add_mlir_library(MLIRAnalysis
  AliasAnalysis.cpp
  CallGraph.cpp
  DataFlowFramework.cpp
  DataLayoutAnalysis.cpp
  FlatLinearValueConstraints.cpp
  Liveness.cpp
  CFGLoopInfo.cpp
  SliceAnalysis.cpp

  AliasAnalysis/LocalAliasAnalysis.cpp

  DataFlow/ConstantPropagationAnalysis.cpp
  DataFlow/DeadCodeAnalysis.cpp
  DataFlow/DenseAnalysis.cpp
  DataFlow/IntegerRangeAnalysis.cpp
  DataFlow/SparseAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRPresburger
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  )

