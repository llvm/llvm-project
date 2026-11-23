add_mlir_conversion_library(MLIRVectorToGPU
  VectorToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToGPU

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAnalysisBase
  MLIRArithDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorToGPUCommonConversion
  )
