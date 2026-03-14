set(LLVM_OPTIONAL_SOURCES
  LLVMIRToLLVMTranslation.cpp
  LLVMToLLVMIRTranslation.cpp
  )

add_mlir_translation_library(MLIRLLVMIRToLLVMTranslation
  LLVMIRToLLVMTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRImport
  )

add_mlir_translation_library(MLIRLLVMToLLVMIRTranslation
  LLVMToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
