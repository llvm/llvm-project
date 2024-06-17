add_mlir_dialect_library(MLIRAMXTransforms
  LegalizeForLLVMExport.cpp
  EnableAMXTileBinding.cpp

  DEPENDS
  MLIRAMXConversionsIncGen

  LINK_LIBS PUBLIC
  MLIRAMXDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
