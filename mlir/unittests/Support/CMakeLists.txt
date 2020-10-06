add_mlir_unittest(MLIRSupportTests
  IndentedOstreamTest.cpp
)

target_link_libraries(MLIRSupportTests
  PRIVATE MLIRSupportIndentedOstream MLIRSupport)
