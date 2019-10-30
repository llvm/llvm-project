
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
  GlobalMappingLayerTest.cpp
  JITTargetMachineBuilderTest.cpp
  LazyCallThroughAndReexportsTest.cpp
  LazyEmittingLayerTest.cpp
  LegacyAPIInteropTest.cpp
  LegacyCompileOnDemandLayerTest.cpp
  LegacyRTDyldObjectLinkingLayerTest.cpp
  ObjectTransformLayerTest.cpp
  OrcCAPITest.cpp
  OrcTestCommon.cpp
  QueueChannel.cpp
  RemoteObjectLayerTest.cpp
  RPCUtilsTest.cpp
  RTDyldObjectLinkingLayerTest.cpp
  SymbolStringPoolTest.cpp
  ThreadSafeModuleTest.cpp
  )

target_link_libraries(OrcJITTests PRIVATE
                        LLVMTestingSupport
                        ${ORC_JIT_TEST_LIBS})
