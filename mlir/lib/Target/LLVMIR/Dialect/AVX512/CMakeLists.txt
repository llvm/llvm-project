add_mlir_translation_library(MLIRAVX512ToLLVMIRTranslation
  AVX512ToLLVMIRTranslation.cpp

  DEPENDS
  MLIRAVX512ConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAVX512
  MLIRLLVMIR
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
