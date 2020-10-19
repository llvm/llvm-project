
set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  Object
  OrcError
  OrcJIT
  Passes
  RuntimeDyld
  Support
  native
  )

add_llvm_unittest(OrcJITTests
  CoreAPIsTest.cpp
  IndirectionUtilsTest.cpp
  JITTargetMachineBuilderTest.cpp
  LazyCallThroughAndReexportsTest.cpp
  OrcTestCommon.cpp
  QueueChannel.cpp
  RPCUtilsTest.cpp
  RTDyldObjectLinkingLayerTest.cpp
  SymbolStringPoolTest.cpp
  ThreadSafeModuleTest.cpp
  )

target_link_libraries(OrcJITTests PRIVATE
                        LLVMTestingSupport
                        ${ORC_JIT_TEST_LIBS})
