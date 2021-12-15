add_subdirectory(Utils)

add_mlir_library(MLIRTransforms
  BufferOptimizations.cpp
  BufferResultsToOutParams.cpp
  BufferUtils.cpp
  Canonicalizer.cpp
  CSE.cpp
  Inliner.cpp
  LocationSnapshot.cpp
  LoopCoalescing.cpp
  LoopFusion.cpp
  LoopInvariantCodeMotion.cpp
  NormalizeMemRefs.cpp
  OpStats.cpp
  ParallelLoopCollapsing.cpp
  PipelineDataTransfer.cpp
  SCCP.cpp
  StripDebugInfo.cpp
  SymbolDCE.cpp
  ViewOpGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRStandardOpsIncGen
  MLIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIRBufferization
  MLIRCopyOpInterface
  MLIRLoopLikeInterface
  MLIRMemRef
  MLIRSCF
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRVector
  )
