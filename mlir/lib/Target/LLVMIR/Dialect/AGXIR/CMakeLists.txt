add_mlir_translation_library(MLIRAGXIRToLLVMIRTranslation
  AGXIRToLLVMIRTranslation.cpp

  DEPENDS
  MLIRAGXIROpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAGXIRDialect
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
