add_mlir_unittest(MLIRSparseTensorTests
  MergerTest.cpp
)
target_link_libraries(MLIRSparseTensorTests
  PRIVATE
  MLIRSparseTensorUtils
)
