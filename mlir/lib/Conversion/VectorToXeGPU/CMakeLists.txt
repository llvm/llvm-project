add_mlir_conversion_library(MLIRVectorToXeGPU
  VectorToXeGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToXeGPU

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorToGPU
  MLIRXeGPUDialect
  MLIRXeGPUUtils
  )
