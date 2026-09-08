add_mlir_conversion_library(MLIRArithToAMDGPU
  ArithToAMDGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToAMDGPU

  LINK_COMPONENTS
  Core

  # ArithToAMDGPU.cpp includes VectorRewritePatterns.h.
  HEADER_LIBS
  MLIRVectorTransforms

  LINK_LIBS PUBLIC
  MLIRAMDGPUDialect
  MLIRAMDGPUUtils
  MLIRArithDialect
  MLIRArithUtils
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
  )
