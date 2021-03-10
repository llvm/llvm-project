add_mlir_dialect_library(MLIRAVX512Transforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  MLIRAVX512ConversionsIncGen

  LINK_LIBS PUBLIC
  MLIRAVX512
  MLIRIR
  MLIRLLVMIR
  )
