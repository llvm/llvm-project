add_llvm_library(MLIRTransformUtils
  FoldUtils.cpp
  GreedyPatternRewriteDriver.cpp
  InliningUtils.cpp
  LoopFusionUtils.cpp
  LoopUtils.cpp
  RegionUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  LINK_LIBS
  MLIRAffineOps
  MLIRAnalysis
  MLIRIR
  MLIRLoopAnalysis
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  LLVMSupport
  )
add_dependencies(MLIRTransformUtils MLIRStandardOpsIncGen)
