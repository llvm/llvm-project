add_subdirectory(Utils)

add_mlir_library(MLIRTransforms
  BufferPlacement.cpp
  Canonicalizer.cpp
  CopyRemoval.cpp
  CSE.cpp
  DialectConversion.cpp
  Inliner.cpp
  LocationSnapshot.cpp
  LoopCoalescing.cpp
  LoopFusion.cpp
  LoopInvariantCodeMotion.cpp
  MemRefDataFlowOpt.cpp
  NormalizeMemRefs.cpp
  OpStats.cpp
  ParallelLoopCollapsing.cpp
  PipelineDataTransfer.cpp
  SCCP.cpp
  StripDebugInfo.cpp
  SymbolDCE.cpp
  ViewOpGraph.cpp
  ViewRegionGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRStandardOpsIncGen
  MLIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIRAnalysis
  MLIRCopyOpInterface
  MLIRLinalgOps
  MLIRLoopLikeInterface
  MLIRSCF
  MLIRPass
  MLIRTransformUtils
  MLIRVector
  )
