add_custom_target(OrcRTUnitTests)
set_target_properties(OrcRTUnitTests PROPERTIES FOLDER "orc-rt/Tests")

if (NOT TARGET llvm_gtest)
  message(WARNING "orc-rt unittests disabled due to GTest being unavailable; "
                  "Try LLVM_INSTALL_GTEST=ON for the LLVM build")
  return ()
endif ()

function(add_orc_rt_unittest test_dirname)
  add_unittest(OrcRTUnitTests ${test_dirname} ${ARGN})
endfunction()

add_orc_rt_unittest(CoreTests
  AllocActionTest.cpp
  BitmaskEnumTest.cpp
  BootstrapInfoTest.cpp
  CallableTraitsHelperTest.cpp
  SimpleSymbolTableTest.cpp
  EndianTest.cpp
  ErrorCAPITest.cpp
  ErrorTest.cpp
  ErrorExceptionInteropTest.cpp
  ExecutorAddressTest.cpp
  ExecutorProcessInfoTest.cpp
  IntervalMapTest.cpp
  IntervalSetTest.cpp
  LockedAccessTest.cpp
  MathTest.cpp
  MemoryAccessSPSCITest.cpp
  MemoryFlagsTest.cpp
  NativeDylibManagerTest.cpp
  NativeDylibManagerSPSCITest.cpp
  QueueingTaskDispatcherTest.cpp
  RTTITest.cpp
  ScopeExitTest.cpp
  SessionTest.cpp
  SimpleNativeMemoryMapTest.cpp
  SimpleNativeMemoryMapSPSCITest.cpp
  SimplePackedSerializationTest.cpp
  SPSAllocActionTest.cpp
  SPSMemoryFlagsTest.cpp
  SPSWrapperFunctionTest.cpp
  SPSWrapperFunctionBufferTest.cpp
  TaskGroupTest.cpp
  ThreadPoolTaskDispatcherTest.cpp
  WrapperFunctionBufferTest.cpp
  bind-test.cpp
  bit-test.cpp
  iterator_range-test.cpp
  move_only_function-test.cpp
  span-test.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB
  )
target_compile_options(CoreTests PRIVATE ${ORC_RT_COMPILE_FLAGS})
target_link_libraries(CoreTests PRIVATE orc-rt-executor)

# Build a shared library for NativeDylibManager tests.
add_library(NativeDylibManagerTestLib SHARED
  Inputs/NativeDylibManagerTestLib.cpp)
set_target_properties(NativeDylibManagerTestLib PROPERTIES
  PREFIX "")
target_compile_definitions(CoreTests PRIVATE
  "NDM_TEST_LIB_PATH=\"$<TARGET_FILE:NativeDylibManagerTestLib>\"")
add_dependencies(CoreTests NativeDylibManagerTestLib)
