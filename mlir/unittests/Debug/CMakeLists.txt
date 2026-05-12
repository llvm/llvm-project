add_mlir_unittest(MLIRDebugTests
  ActionHandlerTest.cpp
  DebugCounterTest.cpp
  ExecutionContextTest.cpp
  FileLineColLocBreakpointManagerTest.cpp
)

mlir_target_link_libraries(MLIRDebugTests
  PRIVATE MLIRDebug)
