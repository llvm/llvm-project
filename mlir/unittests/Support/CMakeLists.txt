add_mlir_unittest(MLIRSupportTests
  IndentedOstreamTest.cpp
  MathExtrasTest.cpp
  StorageUniquerTest.cpp
)

target_link_libraries(MLIRSupportTests
  PRIVATE MLIRSupport)
