add_mlir_translation_library(MLIRX86VectorToLLVMIRTranslation
  X86VectorToLLVMIRTranslation.cpp

  DEPENDS
  MLIRX86VectorConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRX86VectorDialect
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
