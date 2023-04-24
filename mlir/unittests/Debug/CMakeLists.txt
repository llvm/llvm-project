add_mlir_unittest(MLIRDebugTests
  DebugCounterTest.cpp
  ExecutionContextTest.cpp
)

target_link_libraries(MLIRDebugTests
  PRIVATE MLIRDebug)
