add_mlir_translation_library(MLIRBuiltinToLLVMIRTranslation
  BuiltinToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIRCore
  MLIRLLVMDialect
  MLIRSupportCore
  MLIRTargetLLVMIRExport
  )
