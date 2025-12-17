add_mlir_translation_library(MLIRSPIRVToLLVMIRTranslation
  SPIRVToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSPIRVDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
