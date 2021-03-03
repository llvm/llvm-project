add_mlir_translation_library(MLIRLLVMToLLVMIRTranslation
  LLVMToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
