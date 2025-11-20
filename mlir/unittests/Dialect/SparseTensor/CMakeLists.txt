add_mlir_unittest(MLIRSparseTensorTests
  MergerTest.cpp
)
mlir_target_link_libraries(MLIRSparseTensorTests
  PRIVATE
  MLIRSparseTensorUtils
)
