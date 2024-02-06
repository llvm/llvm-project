add_mlir_unittest(MLIRArmSMETests
  TileTypeConversionTest.cpp)
target_link_libraries(MLIRArmSMETests
  PRIVATE
  MLIRArmSMEToLLVM)
