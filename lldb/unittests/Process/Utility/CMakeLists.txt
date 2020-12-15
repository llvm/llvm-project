set(NETBSD_SOURCES
  RegisterContextNetBSDTest_i386.cpp
  RegisterContextNetBSDTest_x86_64.cpp)

if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
  list(APPEND PLATFORM_SOURCES ${NETBSD_SOURCES})
endif()

set(LLVM_OPTIONAL_SOURCES
  ${NETBSD_SOURCES})

add_lldb_unittest(ProcessUtilityTests
  RegisterContextTest.cpp
  RegisterContextFreeBSDTest.cpp
  LinuxProcMapsTest.cpp
  ${PLATFORM_SOURCES}

  LINK_LIBS
    lldbPluginProcessUtility)
