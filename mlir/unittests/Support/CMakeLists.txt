add_mlir_unittest(MLIRSupportTests
  IndentedOstreamTest.cpp
  MathExtrasTest.cpp
)

target_link_libraries(MLIRSupportTests
  PRIVATE MLIRSupportIndentedOstream MLIRSupport)
