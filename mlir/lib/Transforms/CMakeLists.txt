add_subdirectory(Utils)

add_llvm_library(MLIRTransforms
  AffineDataCopyGeneration.cpp
  AffineLoopInvariantCodeMotion.cpp
  Canonicalizer.cpp
  CSE.cpp
  DialectConversion.cpp
  Inliner.cpp
  LocationSnapshot.cpp
  LoopCoalescing.cpp
  LoopFusion.cpp
  LoopInvariantCodeMotion.cpp
  LoopTiling.cpp
  LoopUnrollAndJam.cpp
  LoopUnroll.cpp
  MemRefDataFlowOpt.cpp
  OpStats.cpp
  PipelineDataTransfer.cpp
  SimplifyAffineStructures.cpp
  StripDebugInfo.cpp
  SymbolDCE.cpp
  Vectorize.cpp
  ViewOpGraph.cpp
  ViewRegionGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRLoopLikeInterfaceIncGen
  MLIRStandardOpsIncGen

  LINK_LIBS
  MLIRAffineOps
  MLIRAnalysis
  MLIRIR
  MLIRLoopOps
  MLIRLoopAnalysis
  MLIRPass
  MLIRSupport
  MLIRStandardOps
  MLIRTransformUtils
  MLIRVectorOps
  LLVMSupport
  )
