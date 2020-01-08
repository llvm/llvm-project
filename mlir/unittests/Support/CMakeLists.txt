add_mlir_unittest(MLIRSupportTests
  IndexedAccessorTest.cpp
)

target_link_libraries(MLIRSupportTests
  PRIVATE MLIRSupport)
