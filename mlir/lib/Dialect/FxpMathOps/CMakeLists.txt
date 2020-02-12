add_llvm_library(MLIRFxpMathOps
  IR/FxpMathOps.cpp
  Transforms/LowerUniformRealMath.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/FxpMathOps
  )

target_link_libraries(MLIRFxpMathOps
  MLIRQuantOps
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRStandardOps
  )
add_dependencies(MLIRFxpMathOps
                 MLIRFxpMathOpsIncGen
                 MLIRQuantOps
                 MLIRIR
                 MLIRPass
                 MLIRSupport
                 MLIRStandardOps)
