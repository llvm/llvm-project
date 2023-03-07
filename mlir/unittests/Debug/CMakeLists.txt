add_mlir_unittest(MLIRDebugTests
  DebugCounterTest.cpp
)

target_link_libraries(MLIRDebugTests
  PRIVATE MLIRDebug)
