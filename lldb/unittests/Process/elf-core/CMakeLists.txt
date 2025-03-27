include(CheckSymbolExists)
include(CMakePushCheckState)

add_lldb_unittest(ProcessElfCoreTests
  ThreadElfCoreTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbUtilityHelpers
    lldbPluginProcessElfCore
    lldbPluginPlatformLinux

    LLVMTestingSupport

  LINK_COMPONENTS
    Support
  )

cmake_push_check_state()
set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
check_symbol_exists(gettid "unistd.h" HAVE_GETTID)
if(HAVE_GETTID)
  target_compile_definitions(ProcessElfCoreTests PRIVATE HAVE_GETTID)
endif()
cmake_pop_check_state()
