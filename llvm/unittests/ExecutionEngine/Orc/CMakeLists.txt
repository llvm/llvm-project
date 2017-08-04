
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
  IndirectionUtilsTest.cpp
  GlobalMappingLayerTest.cpp
  LazyEmittingLayerTest.cpp
  ObjectTransformLayerTest.cpp
  OrcCAPITest.cpp
  OrcTestCommon.cpp
  QueueChannel.cpp
  RPCUtilsTest.cpp
  RTDyldObjectLinkingLayerTest.cpp
  )

target_link_libraries(OrcJITTests ${LLVM_PTHREAD_LIB})
