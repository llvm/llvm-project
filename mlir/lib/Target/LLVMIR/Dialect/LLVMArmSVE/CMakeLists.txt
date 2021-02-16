add_mlir_translation_library(MLIRLLVMArmSVEToLLVMIRTranslation
  LLVMArmSVEToLLVMIRTranslation.cpp

  DEPENDS
  MLIRLLVMArmSVEConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMArmSVE
  MLIRLLVMIR
  MLIRSupport
  MLIRTargetLLVMIRModuleTranslation
  )
