add_mlir_dialect_library(MLIRAMXTransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  MLIRAMXConversionsIncGen

  LINK_LIBS PUBLIC
  MLIRAMX
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  )
