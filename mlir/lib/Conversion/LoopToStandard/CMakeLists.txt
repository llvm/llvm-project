add_mlir_conversion_library(MLIRLoopToStandard
  LoopToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LoopToStandard

  DEPENDS
  MLIRConversionPassIncGen
)
target_link_libraries(
  MLIRLoopToStandard
  PUBLIC
  MLIRLoopOps
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )
