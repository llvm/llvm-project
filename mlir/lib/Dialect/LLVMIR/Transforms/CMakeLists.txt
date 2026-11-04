add_mlir_dialect_library(MLIRLLVMIRTransforms
  AddComdats.cpp
  DIExpressionLegalization.cpp
  DIExpressionRewriter.cpp
  DIScopeForLLVMFuncOp.cpp
  InlinerInterfaceImpl.cpp
  LegalizeForExport.cpp
  RequestCWrappers.cpp
  UseDefaultVisibilityPass.cpp

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysisBase
  MLIRIR
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
  )
