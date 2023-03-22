add_mlir_translation_library(MLIRGPUToLLVMIRTranslation
  GPUToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRGPUOps
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
