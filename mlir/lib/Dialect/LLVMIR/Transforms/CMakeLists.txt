add_mlir_dialect_library(MLIRLLVMIRTransforms
  LegalizeForExport.cpp
  OptimizeForNVVM.cpp

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
  MLIRNVVMDialect
  )
