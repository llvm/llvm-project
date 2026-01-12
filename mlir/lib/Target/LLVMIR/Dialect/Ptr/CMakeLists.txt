add_mlir_translation_library(MLIRPtrToLLVMIRTranslation
  PtrToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPtrDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
