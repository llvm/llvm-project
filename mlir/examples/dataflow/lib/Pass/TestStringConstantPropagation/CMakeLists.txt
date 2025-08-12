add_mlir_example_library(MLIRTestStringConstantPropagation
  TestStringConstantPropagation.cpp

  LINK_LIBS PUBLIC
  MLIRStringConstantPropagation
)
