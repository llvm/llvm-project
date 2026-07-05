add_mlir_translation_library(MLIRGPUToLLVMIRTranslation
  GPUToLLVMIRTranslation.cpp
  SelectObjectAttr.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
