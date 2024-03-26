set(LLVM_OPTIONAL_SOURCES
  LLVMIRToPtrTranslation.cpp
  PtrToLLVMIRTranslation.cpp
  )

add_mlir_translation_library(MLIRLLVMIRToPtrTranslation
  LLVMIRToPtrTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPtrDialect
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRImport
  )

add_mlir_translation_library(MLIRPtrToLLVMIRTranslation
  PtrToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPtrDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )