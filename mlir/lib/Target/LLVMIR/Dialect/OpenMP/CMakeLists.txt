add_mlir_translation_library(MLIROpenMPToLLVMIRTranslation
  OpenMPToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIRCore
  MLIRLLVMDialect
  MLIROpenMPDialect
  MLIROpenMPUtils
  MLIRSupportCore
  MLIRTargetLLVMIRExport
  MLIRRewriteUtils
  )
