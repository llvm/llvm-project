add_mlir_translation_library(MLIRTestLLVMTypeTranslation
  TestLLVMTypeTranslation.cpp

  LINK_COMPONENTS
  Core
  TransformUtils

  LINK_LIBS PUBLIC
  MLIRLLVMIR
  MLIRTargetLLVMIRModuleTranslation
  MLIRTestIR
  MLIRTranslation
  )
