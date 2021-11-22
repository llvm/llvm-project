# Main API shared library.
add_mlir_upstream_c_api_library(MLIRCEXECUTIONENGINE
  ExecutionEngine.cpp

  LINK_LIBS PUBLIC
  MLIRExecutionEngine
  MLIRLLVMToLLVMIRTranslation
)
