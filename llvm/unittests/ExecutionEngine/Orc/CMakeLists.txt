
set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  Object
  OrcJIT
  RuntimeDyld
  Support
  native
  )

add_llvm_unittest(OrcJITTests
  CompileOnDemandLayerTest.cpp
  CoreAPIsTest.cpp
  IndirectionUtilsTest.cpp
  GlobalMappingLayerTest.cpp
  LazyEmittingLayerTest.cpp
  LegacyAPIInteropTest.cpp
  ObjectTransformLayerTest.cpp
  OrcCAPITest.cpp
  OrcTestCommon.cpp
  QueueChannel.cpp
  RemoteObjectLayerTest.cpp
  RPCUtilsTest.cpp
  RTDyldObjectLinkingLayerTest.cpp
  RTDyldObjectLinkingLayer2Test.cpp
  SymbolStringPoolTest.cpp
  )

target_link_libraries(OrcJITTests PRIVATE ${ORC_JIT_TEST_LIBS})
