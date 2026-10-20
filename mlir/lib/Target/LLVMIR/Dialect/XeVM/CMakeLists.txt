set(LLVM_OPTIONAL_SOURCES
  XeVMToLLVMIRTranslation.cpp
)

add_mlir_translation_library(MLIRXeVMToLLVMIRTranslation
  XeVMToLLVMIRTranslation.cpp

  DEPENDS
  MLIRXeVMConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRLLVMDialect
  MLIRXeVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
)
