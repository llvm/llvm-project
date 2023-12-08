add_mlir_translation_library(MLIRROCDLToLLVMIRTranslation
  ROCDLToLLVMIRTranslation.cpp

  DEPENDS
  MLIRROCDLConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRROCDLDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
