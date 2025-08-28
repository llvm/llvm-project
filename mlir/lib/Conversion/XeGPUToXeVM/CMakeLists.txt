add_mlir_conversion_library(MLIRXeGPUToXeVM
  XeGPUToXeVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/XeGPUToXeVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRXeVMDialect
  MLIRVectorDialect
  MLIRArithDialect
  MLIRIndexDialect
  MLIRXeGPUDialect
  MLIRPass
  MLIRTransforms
)
