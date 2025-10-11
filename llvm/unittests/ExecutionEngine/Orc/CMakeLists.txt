
set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  IRReader
  JITLink
  Object
  OrcDebugging
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
  LibraryResolverTest.cpp
  LookupAndRecordAddrsTest.cpp
  MachOPlatformTest.cpp
  MapperJITLinkMemoryManagerTest.cpp
  MemoryFlagsTest.cpp
  MemoryMapperTest.cpp
  ObjectFormatsTest.cpp
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
  JITLinkRedirectionManagerTest.cpp
  ReOptimizeLayerTest.cpp

  EXPORT_SYMBOLS
  )

target_link_libraries(OrcJITTests PRIVATE
                        LLVMTestingSupport
                        ${ORC_JIT_TEST_LIBS})
