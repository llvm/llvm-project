add_mlir_translation_library(MLIROpenMPToLLVMIRTranslation
  OpenMPToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysisBase
  MLIRIR
  MLIRLLVMDialect
  MLIROpenMPDialect
  MLIROpenMPUtils
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTransformUtils
  )
