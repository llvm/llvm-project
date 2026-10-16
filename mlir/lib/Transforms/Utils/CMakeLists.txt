# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIRRewriteUtils
  OBJECT
  CSE.cpp
  DialectConversion.cpp
  FoldUtils.cpp
  GreedyPatternRewriteDriver.cpp
  RegionUtils.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRTransformsDialectInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysisUtilities
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRRewrite
  MLIRPass

  HEADER_LIBS
  MLIRSubsetOpInterface
  )

add_mlir_library(MLIRTransformUtils
  CFGToSCF.cpp
  CommutativityUtils.cpp
  ControlFlowSinkUtils.cpp
  Inliner.cpp
  InliningUtils.cpp
  LoopInvariantCodeMotionUtils.cpp
  WalkPatternRewriteDriver.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRTransformsDialectInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRRewriteUtils
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRSubsetOpInterface
  MLIRRewrite
  MLIRPass
  )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRTransformUtils PRIVATE $<TARGET_OBJECTS:obj.MLIRRewriteUtils>)
endif()
