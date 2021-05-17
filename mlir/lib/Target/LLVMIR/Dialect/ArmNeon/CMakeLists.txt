add_mlir_translation_library(MLIRArmNeonToLLVMIRTranslation
  ArmNeonToLLVMIRTranslation.cpp

  DEPENDS
  MLIRArmNeonConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArmNeon
  MLIRIR
  MLIRLLVMIR
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
