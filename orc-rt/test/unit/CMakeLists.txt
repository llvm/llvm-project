add_custom_target(OrcRTUnitTests)
set_target_properties(OrcRTUnitTests PROPERTIES FOLDER "orc-rt/Tests")

if (NOT TARGET llvm_gtest)
  message(WARNING "orc-rt unittests disabled due to GTest being unavailable; "
                  "Try LLVM_INSTALL_GTEST=ON for the LLVM build")
  return ()
endif ()

# Adds a unit test suite for one layer. link_lib is the layer's library; the
# remaining arguments are sources.
function(add_orc_rt_unittest test_name link_lib)
  add_unittest(OrcRTUnitTests ${test_name} ${ARGN} DISABLE_LLVM_LINK_LLVM_DYLIB)
  target_compile_options(${test_name} PRIVATE ${ORC_RT_COMPILE_FLAGS})
  # Shared test helpers (CommonTestUtils.h, DirectCaller.h, ...) live at the
  # root of this directory, so tests in the layer subdirectories can include
  # them by bare name.
  target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${test_name} PRIVATE ${link_lib})
endfunction()

# SupportTests links the Support objects and nothing else, so a new
# support-to-bedrock dependency shows up here as a link error rather than
# going unnoticed.
add_orc_rt_unittest(SupportTests orc-rt-support
  support/AllocActionTest.cpp
  support/BitmaskEnumTest.cpp
  support/CAPICompileTest.c
  support/CCallbackTest.cpp
  support/CallableTraitsHelperTest.cpp
  support/CompilerTest.cpp
  support/EndianTest.cpp
  support/ErrorCAPITest.cpp
  support/ErrorExceptionInteropTest.cpp
  support/ErrorTest.cpp
  support/ExecutorAddressTest.cpp
  support/IntervalMapTest.cpp
  support/IntervalSetTest.cpp
  support/LockedAccessTest.cpp
  support/LoggingTest.cpp
  support/MacroUtilsTest.cpp
  support/MemoryFlagsTest.cpp
  support/ProxyTest.cpp
  support/RTTITest.cpp
  support/StringExtrasTest.cpp
  support/StringPoolTest.cpp
  support/WrapperFunctionBufferTest.cpp
  support/bind-test.cpp
  support/bit-test.cpp
  support/iterator_range-test.cpp
  support/move_only_function-test.cpp
  support/scope_exit-test.cpp
  support/span-test.cpp

  support/sps/SPSAllocActionTest.cpp
  support/sps/SPSMemoryFlagsTest.cpp
  support/sps/SPSWrapperFunctionBufferTest.cpp
  support/sps/SPSWrapperFunctionTest.cpp
  support/sps/SimplePackedSerializationTest.cpp

  support/sys/ErrnoTest.cpp

  tools/OptionParserTest.cpp
  )

add_orc_rt_unittest(BedrockTests orc-rt-bedrock
  bedrock/BootstrapInfoTest.cpp
  bedrock/ExecutorProcessInfoTest.cpp
  bedrock/InProcessControllerAccessTest.cpp
  bedrock/NativeDylibManagerTest.cpp
  bedrock/QueueingRunnerTest.cpp
  bedrock/SessionTest.cpp
  bedrock/SimpleNativeMemoryMapTest.cpp
  bedrock/SimpleSymbolTableTest.cpp
  bedrock/StandaloneMachOUnwindInfoRegistrarTest.cpp
  bedrock/TaskGroupTest.cpp
  bedrock/ThreadPoolRunnerTest.cpp

  bedrock/sps/CallSPSCITest.cpp
  bedrock/sps/MemoryAccessSPSCITest.cpp
  bedrock/sps/NativeDylibManagerSPSCITest.cpp
  bedrock/sps/SimpleNativeMemoryMapSPSCITest.cpp
  bedrock/sps/SimpleRemoteCATest.cpp

  bedrock/sys/CPUFeaturesTest.cpp
  bedrock/sys/TargetTripleTest.cpp
  )

# Build a shared library for NativeDylibManager tests.
add_library(NativeDylibManagerTestLib SHARED
  Inputs/NativeDylibManagerTestLib.cpp)
set_target_properties(NativeDylibManagerTestLib PROPERTIES
  PREFIX "")
target_compile_definitions(BedrockTests PRIVATE
  "NDM_TEST_LIB_PATH=\"$<TARGET_FILE:NativeDylibManagerTestLib>\"")
add_dependencies(BedrockTests NativeDylibManagerTestLib)
