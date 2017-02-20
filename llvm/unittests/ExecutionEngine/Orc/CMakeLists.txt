
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
  RTDyldObjectLinkingLayerTest.cpp
  ObjectTransformLayerTest.cpp
  OrcCAPITest.cpp
  OrcTestCommon.cpp
  RPCUtilsTest.cpp
  )

target_link_libraries(OrcJITTests ${LLVM_PTHREAD_LIB})
