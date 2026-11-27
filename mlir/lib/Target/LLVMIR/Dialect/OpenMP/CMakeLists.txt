add_mlir_translation_library(MLIROpenMPToLLVMIRTranslation
  OpenMPCommon.cpp
  OpenMPToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core
  FrontendOpenMP

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
