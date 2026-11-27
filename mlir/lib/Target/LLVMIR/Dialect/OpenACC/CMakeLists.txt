add_mlir_translation_library(MLIROpenACCToLLVMIRTranslation
  OpenACCToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  BinaryFormat
  Core
  FrontendOffloading
  Object
  TransformUtils

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIROpenACCDialect
  MLIRROCDLDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRXeVMDialect
  )
