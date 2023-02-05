
set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  IRReader
  JITLink
  Object
  OrcJIT
  OrcShared
  OrcTargetProcess
  Passes
  RuntimeDyld
  Support
  TargetParser
  native
  )

add_llvm_unittest(OrcJITTests
  CoreAPIsTest.cpp
  ExecutorAddressTest.cpp
  ExecutionSessionWrapperFunctionCallsTest.cpp
  EPCGenericJITLinkMemoryManagerTest.cpp
  EPCGenericMemoryAccessTest.cpp
  IndirectionUtilsTest.cpp
  JITTargetMachineBuilderTest.cpp
  LazyCallThroughAndReexportsTest.cpp
  LookupAndRecordAddrsTest.cpp
  MapperJITLinkMemoryManagerTest.cpp
  MemoryMapperTest.cpp
  ObjectLinkingLayerTest.cpp
  OrcCAPITest.cpp
  OrcTestCommon.cpp
  ResourceTrackerTest.cpp
  RTDyldObjectLinkingLayerTest.cpp
  SharedMemoryMapperTest.cpp
  SimpleExecutorMemoryManagerTest.cpp
  SimplePackedSerializationTest.cpp
  SymbolStringPoolTest.cpp
  TaskDispatchTest.cpp
  ThreadSafeModuleTest.cpp
  WrapperFunctionUtilsTest.cpp
  )

target_link_libraries(OrcJITTests PRIVATE
                        LLVMTestingSupport
                        ${ORC_JIT_TEST_LIBS})

set_property(TARGET OrcJITTests PROPERTY FOLDER "Tests/UnitTests/ExecutionTests")

export_executable_symbols(OrcJITTests)
