add_mlir_conversion_library(MLIRCombToSMT
  ArithToSMT.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRSMT
  MLIRTransforms
)
