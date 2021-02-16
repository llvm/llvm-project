add_mlir_translation_library(MLIRLLVMAVX512ToLLVMIRTranslation
  LLVMAVX512ToLLVMIRTranslation.cpp

  DEPENDS
  MLIRLLVMAVX512ConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMAVX512
  MLIRLLVMIR
  MLIRSupport
  MLIRTargetLLVMIRModuleTranslation
  )
