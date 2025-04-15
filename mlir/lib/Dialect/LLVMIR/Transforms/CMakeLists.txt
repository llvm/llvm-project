add_mlir_dialect_library(MLIRLLVMIRTransforms
  AddComdats.cpp
  DIExpressionLegalization.cpp
  DIExpressionRewriter.cpp
  DIScopeForLLVMFuncOp.cpp
  InlinerInterfaceImpl.cpp
  LegalizeForExport.cpp
  OptimizeForNVVM.cpp
  RequestCWrappers.cpp

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
  MLIRNVVMDialect
  )
