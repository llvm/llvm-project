add_mlir_conversion_library(MLIRLLVMToControlFlow
  LLVMToControlFlow.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LLVMToControlFlow

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRControlFlowDialect
  MLIRLLVMDialect
  MLIRTransforms
  )

