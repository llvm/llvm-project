add_mlir_translation_library(MLIRArmNeonToLLVMIRTranslation
  ArmNeonToLLVMIRTranslation.cpp

  DEPENDS
  MLIRArmNeonConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArmNeonDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
