add_mlir_unittest(MLIRAMDGPUTests
  AMDGPUUtilsTest.cpp
)
mlir_target_link_libraries(MLIRAMDGPUTests
  PRIVATE
  MLIRAMDGPUUtils
)
