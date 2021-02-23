add_mlir_unittest(MLIRSupportTests
  DebugActionTest.cpp
  IndentedOstreamTest.cpp
  MathExtrasTest.cpp
)

target_link_libraries(MLIRSupportTests
  PRIVATE MLIRSupportIndentedOstream MLIRSupport)
