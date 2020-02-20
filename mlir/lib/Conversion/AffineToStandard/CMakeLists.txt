add_mlir_conversion_library(MLIRAffineToStandard
  AffineToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AffineToStandard
)
target_link_libraries(
  MLIRAffineToStandard
  PUBLIC
  MLIRAffineOps
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  MLIRTransforms
  MLIRIR
  LLVMCore
  LLVMSupport
  )
