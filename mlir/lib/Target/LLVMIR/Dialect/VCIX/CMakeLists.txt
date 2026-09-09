add_mlir_translation_library(MLIRVCIXToLLVMIRTranslation
  VCIXToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRVCIXDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
