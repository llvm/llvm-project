add_mlir_upstream_c_api_library(MLIRCAPITarget
  LLVMIR.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRToLLVMIRTranslationRegistration
  MLIRCAPIIR
  MLIRLLVMToLLVMIRTranslation
  MLIRLLVMIRToLLVMTranslation
  MLIRSupport
)
