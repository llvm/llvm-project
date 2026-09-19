add_custom_target(OrcRTUnitTests)
set_target_properties(OrcRTUnitTests PROPERTIES
  FOLDER "orc-rt/Tests"
)

if(NOT TARGET llvm_gtest)
  message(WARNING
    "orc-rt unittests disabled due to GTest being unavailable; "
    "try LLVM_INSTALL_GTEST=ON for the LLVM build")
  return()
endif()

# orc-rt tests exercise RTTI and exception handling explicitly.
#
# Set these before creating any unit-test targets so LLVM's test infrastructure
# does not add /GR- or exception-disabling flags and then override them later.
set(LLVM_REQUIRES_RTTI ON)
set(LLVM_REQUIRES_EH ON)

# Add an orc-rt unit-test suite.
#
# SOURCES
#   Sources that are always part of the suite. Platform-specific sources are
#   added afterwards with target_sources().
#
# LINK_LIBS
#   Runtime object libraries required by the suite.
function(add_orc_rt_unittest test_name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG
    ""
    ""
    "SOURCES;LINK_LIBS"
  )

  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unknown arguments to add_orc_rt_unittest(${test_name}): "
      "${ARG_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT ARG_SOURCES)
    message(FATAL_ERROR
      "add_orc_rt_unittest(${test_name}) requires SOURCES")
  endif()

  add_unittest(OrcRTUnitTests ${test_name}
    ${ARG_SOURCES}
    DISABLE_LLVM_LINK_LLVM_DYLIB
  )

  target_compile_options(${test_name}
    PRIVATE
      ${ORC_RT_COMPILE_FLAGS}
  )

  # Shared test helpers (CommonTestUtils.h, DirectCaller.h, ...) live at the
  # root of this directory, so tests in subdirectories can include them by
  # bare name.
  target_include_directories(${test_name}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )

  if(ARG_LINK_LIBS)
    target_link_libraries(${test_name}
      PRIVATE
        ${ARG_LINK_LIBS}
    )
  endif()
endfunction()


#==============================================================================
# Support tests
#==============================================================================

# SupportTests deliberately links only the Support objects. If Support acquires
# an accidental dependency on Bedrock it should show up here as a link failure.
add_orc_rt_unittest(SupportTests
  SOURCES
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
    support/ManglingTest.cpp
    support/MemoryFlagsTest.cpp
    support/ProxyTest.cpp
    support/RTTICrossDylibTest.cpp
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

    tools/OptionParserTest.cpp

  LINK_LIBS
    orc-rt-support-objects
)

# Errno.h currently has a POSIX implementation only.
if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_sources(SupportTests
    PRIVATE
      support/sys/ErrnoTest.cpp
  )
endif()


#==============================================================================
# Bedrock tests
#==============================================================================

add_orc_rt_unittest(BedrockTests
  SOURCES
    bedrock/BootstrapInfoTest.cpp
    bedrock/ExecutorProcessInfoTest.cpp
    bedrock/InProcessControllerAccessTest.cpp
    bedrock/NativeDylibManagerTest.cpp
    bedrock/QueueingRunnerTest.cpp
    bedrock/SessionTest.cpp
    bedrock/SimpleNativeMemoryMapTest.cpp
    bedrock/SimpleSymbolTableTest.cpp
    bedrock/TaskGroupTest.cpp
    bedrock/ThreadPoolRunnerTest.cpp

    bedrock/sps/CallSPSCITest.cpp
    bedrock/sps/MemoryAccessSPSCITest.cpp
    bedrock/sps/NativeDylibManagerSPSCITest.cpp
    bedrock/sps/SimpleNativeMemoryMapSPSCITest.cpp
    bedrock/sps/SimpleRemoteCATest.cpp

    bedrock/sys/CPUFeaturesTest.cpp
    bedrock/sys/TargetTripleTest.cpp

  LINK_LIBS
    orc-rt-bedrock-objects
    orc-rt-support-objects
)

# POSIX-specific test support.
#
# SocketHandleTest uses SocketTestUtils, for which there is currently only a
# POSIX implementation.
if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_sources(BedrockTests
    PRIVATE
      bedrock/SocketHandleTest.cpp
      bedrock/sys/posix/SocketTestUtils.cpp
  )
endif()

# Mach-O unwind registration is Darwin-specific.
if(APPLE)
  target_sources(BedrockTests
    PRIVATE
      bedrock/StandaloneMachOUnwindInfoRegistrarTest.cpp
  )
endif()


#==============================================================================
# NativeDylibManager test library
#==============================================================================

# A real shared library used by NativeDylibManagerTest.
add_library(NativeDylibManagerTestLib SHARED
  Inputs/NativeDylibManagerTestLib.cpp
)

set_target_properties(NativeDylibManagerTestLib PROPERTIES
  PREFIX ""
  FOLDER "orc-rt/Tests"
)

target_compile_options(NativeDylibManagerTestLib
  PRIVATE
    ${ORC_RT_COMPILE_FLAGS}
)

target_compile_definitions(BedrockTests
  PRIVATE
    "NDM_TEST_LIB_PATH=\"$<TARGET_FILE:NativeDylibManagerTestLib>\""
)

add_dependencies(BedrockTests
  NativeDylibManagerTestLib
)


#==============================================================================
# Cross-dylib RTTI test library
#==============================================================================

# RTTICrossDylibTestLib links its own copy of the runtime objects, giving it a
# LibraryID distinct from SupportTests's. This exercises RTTI identity across a
# real shared-library boundary rather than simulating one inside one binary.
add_library(RTTICrossDylibTestLib SHARED
  Inputs/RTTICrossDylibTestLib.cpp
)

set_target_properties(RTTICrossDylibTestLib PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "orc-rt/Tests"
)

target_compile_options(RTTICrossDylibTestLib
  PRIVATE
    ${ORC_RT_COMPILE_FLAGS}
)

target_link_libraries(RTTICrossDylibTestLib
  PRIVATE
    orc-rt-bedrock-objects
    orc-rt-support-objects
)

target_link_libraries(SupportTests
  PRIVATE
    RTTICrossDylibTestLib
)
