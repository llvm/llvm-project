add_mlir_unittest(MLIRAGXTests
  ChipInfoTest.cpp
)
mlir_target_link_libraries(MLIRAGXTests
  PRIVATE
  MLIRAGXUtils
)
