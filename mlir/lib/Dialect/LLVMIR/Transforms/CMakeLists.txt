add_mlir_dialect_library(MLIRLLVMIRTransforms
  LegalizeForExport.cpp

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  )
