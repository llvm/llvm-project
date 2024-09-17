add_mlir_unittest(MLIRAMDGPUTests
  AMDGPUUtilsTest.cpp
)
target_link_libraries(MLIRAMDGPUTests
  PRIVATE
  MLIRAMDGPUUtils
)
