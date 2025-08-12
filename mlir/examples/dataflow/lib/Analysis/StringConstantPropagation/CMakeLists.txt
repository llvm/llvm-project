add_mlir_example_library(MLIRStringConstantPropagation
  StringConstantPropagation.cpp

  LINK_LIBS PUBLIC
  MLIRAnalysis
  )
