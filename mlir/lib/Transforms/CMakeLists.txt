add_subdirectory(Utils)

add_mlir_library(MLIRTransforms
  BufferPlacement.cpp
  Canonicalizer.cpp
  CSE.cpp
  DialectConversion.cpp
  Inliner.cpp
  LocationSnapshot.cpp
  LoopCoalescing.cpp
  LoopFusion.cpp
  LoopInvariantCodeMotion.cpp
  MemRefDataFlowOpt.cpp
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
  MLIRLinalgOps
  MLIRLoopLikeInterface
  MLIRSCF
  MLIRPass
  MLIRTransformUtils
  MLIRVector
  )
