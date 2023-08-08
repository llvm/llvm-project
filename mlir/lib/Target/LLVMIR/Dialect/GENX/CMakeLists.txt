add_mlir_translation_library(MLIRGENXToLLVMIRTranslation
  GENXToLLVMIRTranslation.cpp

  DEPENDS
  MLIRGENXConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRGENXDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )