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
  CCallbackTest.cpp
  CallSPSCITest.cpp
  CallableTraitsHelperTest.cpp
  CommandLineTest.cpp
  CompilerTest.cpp
  EndianTest.cpp
  ErrorCAPITest.cpp
  ErrorExceptionInteropTest.cpp
  ErrorTest.cpp
  ExecutorAddressTest.cpp
  ExecutorProcessInfoTest.cpp
  InProcessControllerAccessTest.cpp
  IntervalMapTest.cpp
  IntervalSetTest.cpp
  LockedAccessTest.cpp
  LoggingTest.cpp
  MacroUtilsTest.cpp
  MathTest.cpp
  MemoryAccessSPSCITest.cpp
  MemoryFlagsTest.cpp
  NativeDylibManagerSPSCITest.cpp
  NativeDylibManagerTest.cpp
  QueueingRunnerTest.cpp
  RTTITest.cpp
  SPSAllocActionTest.cpp
  SPSMemoryFlagsTest.cpp
  SPSWrapperFunctionBufferTest.cpp
  SPSWrapperFunctionTest.cpp
  SessionTest.cpp
  SimpleNativeMemoryMapSPSCITest.cpp
  SimpleNativeMemoryMapTest.cpp
  SimplePackedSerializationTest.cpp
  SimpleSymbolTableTest.cpp
  StandaloneMachOUnwindInfoRegistrarTest.cpp
  TaskGroupTest.cpp
  ThreadPoolRunnerTest.cpp
  WrapperFunctionBufferTest.cpp
  bind-test.cpp
  bit-test.cpp
  iterator_range-test.cpp
  move_only_function-test.cpp
  scope_exit-test.cpp
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
