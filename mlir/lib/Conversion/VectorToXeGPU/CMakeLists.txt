add_mlir_conversion_library(MLIRVectorToXeGPU
  VectorToXeGPU.cpp
  WholeBufferPromotion.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToXeGPU

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRMemorySlotInterfaces
  MLIRMemRefDialect
  MLIRTransforms
  MLIRUBDialect
  MLIRVectorDialect
  MLIRVectorToGPU
  MLIRXeGPUDialect
  MLIRXeGPUUtils
  )
