add_mlir_translation_library(MLIRAMXToLLVMIRTranslation
  AMXToLLVMIRTranslation.cpp

  DEPENDS
  MLIRAMXConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAMX
  MLIRLLVMIR
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
