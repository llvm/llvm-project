add_mlir_translation_library(MLIRNVVMToLLVMIRTranslation
  NVVMToLLVMIRTranslation.cpp

  DEPENDS
  MLIRNVVMConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
