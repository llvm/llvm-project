add_mlir_library(MLIRTransformUtils
  FoldUtils.cpp
  GreedyPatternRewriteDriver.cpp
  InliningUtils.cpp
  LoopFusionUtils.cpp
  LoopUtils.cpp
  RegionUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRStandardOpsIncGen
  )

target_link_libraries(MLIRTransformUtils
  PUBLIC
  MLIRAffineOps
  MLIRAnalysis
  MLIRLoopAnalysis
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  )
