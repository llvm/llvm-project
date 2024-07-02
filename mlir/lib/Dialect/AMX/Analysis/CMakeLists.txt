add_mlir_dialect_library(MLIRAMXAnalysis
  AMXBindingAnalysis.cpp

  DEPENDS
  MLIRAMXConversionsIncGen

  LINK_LIBS PUBLIC
  MLIRAMXDialect
  MLIRIR
  )
