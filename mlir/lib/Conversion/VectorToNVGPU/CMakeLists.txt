add_mlir_conversion_library(MLIRVectorToNVGPU
  VectorToNVGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToNVGPU

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAnalysisBase
  MLIRArithDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRNVGPUDialect
  MLIRNVGPUUtils
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorToGPUCommonConversion
  MLIRVectorTransforms
  )
