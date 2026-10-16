set(LLVM_OPTIONAL_SOURCES
  AliasAnalysis.cpp
  CallGraph.cpp
  DataLayoutAnalysis.cpp
  FlatLinearValueConstraints.cpp
  Liveness.cpp
  CFGLoopInfo.cpp
  SliceAnalysis.cpp
  TopologicalSortUtils.cpp

  AliasAnalysis/LocalAliasAnalysis.cpp

  DataFlow/ConstantPropagationAnalysis.cpp
  DataFlow/DeadCodeAnalysis.cpp
  DataFlow/DenseAnalysis.cpp
  DataFlow/IntegerDivisibilityAnalysis.cpp
  DataFlow/IntegerRangeAnalysis.cpp
  DataFlow/LivenessAnalysis.cpp
  DataFlow/SparseAnalysis.cpp
  )

add_subdirectory(Presburger)

# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIRAnalysisUtilities
  OBJECT
  AliasAnalysis.cpp
  DataLayoutAnalysis.cpp
  FlatLinearValueConstraints.cpp
  Liveness.cpp
  SliceAnalysis.cpp
  TopologicalSortUtils.cpp
  AliasAnalysis/LocalAliasAnalysis.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRDialectCoreUtils
  MLIRFunctionInterfaces
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRPresburgerIntegerRelations
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface

  )

add_mlir_library(MLIRAnalysis
  CallGraph.cpp
  DataFlowFramework.cpp
  CFGLoopInfo.cpp
  SliceWalk.cpp

  DataFlow/ConstantPropagationAnalysis.cpp
  DataFlow/DeadCodeAnalysis.cpp
  DataFlow/DenseAnalysis.cpp
  DataFlow/IntegerDivisibilityAnalysis.cpp
  DataFlow/IntegerRangeAnalysis.cpp
  DataFlow/LivenessAnalysis.cpp
  DataFlow/SparseAnalysis.cpp
  DataFlow/StridedMetadataRangeAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  LINK_LIBS PUBLIC
  MLIRAnalysisUtilities
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRDialectUtils
  MLIRFunctionInterfaces
  MLIRInferIntDivisibilityOpInterface
  MLIRInferIntRangeInterface
  MLIRInferStridedMetadataInterface
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRPresburger
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRAnalysis PRIVATE $<TARGET_OBJECTS:obj.MLIRAnalysisUtilities>)
endif()

