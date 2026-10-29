add_mlir_conversion_library(MLIRArithCommonConversion
  CeilFloorDivExpansion.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  )
