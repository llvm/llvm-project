set(LLVM_OPTIONAL_SOURCES
  LLVMIRToNVVMTranslation.cpp
  NVVMToLLVMIRTranslation.cpp
  )

add_mlir_translation_library(MLIRLLVMIRToNVVMTranslation
  LLVMIRToNVVMTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRNVVMDialect
  MLIRSupport
  MLIRTargetLLVMIRImport
  )

add_mlir_translation_library(MLIRNVVMToLLVMIRTranslation
  NVVMToLLVMIRTranslation.cpp

  DEPENDS
  MLIRNVVMConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
