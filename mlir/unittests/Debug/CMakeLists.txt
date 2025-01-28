add_mlir_unittest(MLIRDebugTests
  DebugCounterTest.cpp
  ExecutionContextTest.cpp
  FileLineColLocBreakpointManagerTest.cpp
)

mlir_target_link_libraries(MLIRDebugTests
  PRIVATE MLIRDebug)
