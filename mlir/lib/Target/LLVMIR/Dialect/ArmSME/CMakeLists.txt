add_mlir_translation_library(MLIRArmSMEToLLVMIRTranslation
  ArmSMEToLLVMIRTranslation.cpp

  DEPENDS
  MLIRArmSMEConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArmSMEDialect
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
