add_mlir_unittest(MLIRSupportTests
  DebugActionTest.cpp
  DebugCounterTest.cpp
  IndentedOstreamTest.cpp
  MathExtrasTest.cpp
)

target_link_libraries(MLIRSupportTests
  PRIVATE MLIRSupportIndentedOstream MLIRSupport)
