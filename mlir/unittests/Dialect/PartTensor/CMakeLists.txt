add_mlir_unittest(MLIRPartTensorTests
  PartTensorTest.cpp
)
target_link_libraries(MLIRPartTensorTests
  PRIVATE
  MLIRExecutionEngine
  MLIRSparseTensorUtils
)
