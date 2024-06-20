add_mlir_unittest(MLIRSupportTests
  IndentedOstreamTest.cpp
  StorageUniquerTest.cpp
  ScalableVectorTypeTest.cpp
)

target_link_libraries(MLIRSupportTests
  PRIVATE MLIRSupport MLIRIR)
