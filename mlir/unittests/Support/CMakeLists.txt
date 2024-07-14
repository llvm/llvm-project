add_mlir_unittest(MLIRSupportTests
  CyclicReplacerCacheTest.cpp
  IndentedOstreamTest.cpp
  StorageUniquerTest.cpp
)

target_link_libraries(MLIRSupportTests
  PRIVATE MLIRSupport)
