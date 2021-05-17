add_mlir_translation_library(MLIROpenACCToLLVMIRTranslation
  OpenACCToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIROpenACC
  MLIROpenACCToLLVM
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
