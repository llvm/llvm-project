# Analysis libraries are split by subsystem so clients can link only the APIs
# they use:
#
#   MLIRAnalysisBase       General IR analyses: alias analysis, call graphs,
#                          CFG loop information, data layout, liveness, slices,
#                          and topological sorting.
#   MLIRAnalysisDataFlow   The data-flow framework and its analyses, including
#                          data-flow liveness (distinct from Base's liveness).
#   MLIRAnalysisPresburger IR-aware constraint APIs: FlatLinearConstraints and
#                          FlatLinearValueConstraints, adapting Presburger
#                          mathematics to affine expressions and SSA values.
#   MLIRAnalysis           Umbrella containing all three components.
#
# The three components are independent of one another. Base and DataFlow have
# no Presburger dependency. The IR-aware adapter links MLIRIR and MLIRPresburger;
# the latter, defined in Presburger/, is a generic mathematical library
# with no MLIR IR dependency.
#
# Each implementation is compiled once into its component's object library.
# The umbrella reuses these objects to provide the complete static archive.
# Aggregate libMLIR includes only the components, excluding the umbrella so that
# each defining object appears exactly once.

add_subdirectory(Presburger)

# Keep source ownership separate from the umbrella archive below. Object
# libraries let both archives reuse the same compilations, including in shared
# library builds.
add_mlir_library(MLIRAnalysisBase
  AliasAnalysis.cpp
  AliasAnalysis/LocalAliasAnalysis.cpp
  CallGraph.cpp
  CFGLoopInfo.cpp
  DataLayoutAnalysis.cpp
  Liveness.cpp
  SliceAnalysis.cpp
  SliceWalk.cpp
  TopologicalSortUtils.cpp

  OBJECT
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  )

add_mlir_library(MLIRAnalysisDataFlow
  DataFlowFramework.cpp
  DataFlow/ConstantPropagationAnalysis.cpp
  DataFlow/DeadCodeAnalysis.cpp
  DataFlow/DenseAnalysis.cpp
  DataFlow/IntegerDivisibilityAnalysis.cpp
  DataFlow/IntegerRangeAnalysis.cpp
  DataFlow/LivenessAnalysis.cpp
  DataFlow/SparseAnalysis.cpp
  DataFlow/StridedMetadataRangeAnalysis.cpp

  OBJECT
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDialectUtils
  MLIRInferIntDivisibilityOpInterface
  MLIRInferIntRangeInterface
  MLIRInferStridedMetadataInterface
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  )

# This is the IR-aware adapter; MLIRPresburger is independent of MLIR IR.
add_mlir_library(MLIRAnalysisPresburger
  FlatLinearValueConstraints.cpp

  OBJECT
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPresburger
  )

# Provide the complete archive for users linking libMLIRAnalysis.a directly.
# Only the components participate in libMLIR, so each object is included there
# exactly once. Link the components' dependencies rather than the components
# themselves, since their definitions are already embedded here.
add_mlir_library(MLIRAnalysis
  $<TARGET_OBJECTS:obj.MLIRAnalysisBase>
  $<TARGET_OBJECTS:obj.MLIRAnalysisDataFlow>
  $<TARGET_OBJECTS:obj.MLIRAnalysisPresburger>

  EXCLUDE_FROM_LIBMLIR
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  $<TARGET_PROPERTY:MLIRAnalysisBase,INTERFACE_LINK_LIBRARIES>
  $<TARGET_PROPERTY:MLIRAnalysisDataFlow,INTERFACE_LINK_LIBRARIES>
  $<TARGET_PROPERTY:MLIRAnalysisPresburger,INTERFACE_LINK_LIBRARIES>
  # The umbrella also exposes the type inference interface.
  MLIRInferTypeOpInterface
  )
