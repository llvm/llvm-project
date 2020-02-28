add_mlir_conversion_library(MLIRLoopToStandard
  ConvertLoopToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LoopToStandard

  LINK_LIBS
  MLIRIR
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )
add_dependencies(
  MLIRLoopToStandard

  MLIRLoopOps
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
