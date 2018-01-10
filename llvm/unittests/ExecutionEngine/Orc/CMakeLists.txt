
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
  ObjectTransformLayerTest.cpp
  OrcCAPITest.cpp
  OrcTestCommon.cpp
  QueueChannel.cpp
  RemoteObjectLayerTest.cpp
  RPCUtilsTest.cpp
  RTDyldObjectLinkingLayerTest.cpp
  SymbolStringPoolTest.cpp
  )

set(ORC_JIT_TEST_LIBS ${LLVM_PTHREAD_LIB})
if(NOT HAVE_CXX_ATOMICS64_WITHOUT_LIB)
  list(APPEND ORC_JIT_TEST_LIBS atomic)
endif()

target_link_libraries(OrcJITTests PRIVATE ${ORC_JIT_TEST_LIBS})
