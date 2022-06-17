add_mlir_dialect_library(MLIRLLVMIRTransforms
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
