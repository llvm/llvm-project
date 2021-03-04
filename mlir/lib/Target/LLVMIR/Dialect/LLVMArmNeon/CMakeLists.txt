add_mlir_translation_library(MLIRLLVMArmNeonToLLVMIRTranslation
  LLVMArmNeonToLLVMIRTranslation.cpp

  DEPENDS
  MLIRLLVMArmNeonConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMArmNeon
  MLIRLLVMIR
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
