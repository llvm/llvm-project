# Main API shared library.
add_mlir_upstream_c_api_library(MLIRCAPIExecutionEngine
  ExecutionEngine.cpp

  LINK_LIBS PUBLIC
  MLIRExecutionEngine
  MLIRLLVMToLLVMIRTranslation
)
