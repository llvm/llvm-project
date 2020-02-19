add_llvm_library(MLIRLoopOpsTransforms
  ParallelLoopFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LoopOps
  )

target_link_libraries(MLIRLoopOpsTransforms
  MLIRPass
  MLIRLoopOps
  )
