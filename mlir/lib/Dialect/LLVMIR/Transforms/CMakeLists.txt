add_mlir_dialect_library(MLIRLLVMIRTransforms
  LegalizeForExport.cpp
  OptimizeForNVVM.cpp

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRTransforms
  MLIRNVVMIR
  )
