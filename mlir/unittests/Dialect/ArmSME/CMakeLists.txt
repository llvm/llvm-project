add_mlir_unittest(MLIRArmSMETests
  TileTypeConversionTest.cpp)
mlir_target_link_libraries(MLIRArmSMETests
  PRIVATE
  MLIRArmSMEToLLVM)
