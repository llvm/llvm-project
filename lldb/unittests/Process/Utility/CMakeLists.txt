set(FREEBSD_SOURCES
  RegisterInfoFreeBSDTest.cpp)
set(NETBSD_SOURCES
  RegisterInfoNetBSDTest_i386.cpp
  RegisterInfoNetBSDTest_x86_64.cpp)

if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  list(APPEND PLATFORM_SOURCES ${FREEBSD_SOURCES})
elseif (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
  list(APPEND PLATFORM_SOURCES ${NETBSD_SOURCES})
endif()

set(LLVM_OPTIONAL_SOURCES
  ${FREEBSD_SOURCES}
  ${NETBSD_SOURCES})

add_lldb_unittest(ProcessUtilityTests
  PARTIAL_SOURCES_INTENDED
  LinuxProcMapsTest.cpp
  MemoryTagManagerAArch64MTETest.cpp
  CoreFileMemoryRangesTest.cpp

  LINK_LIBS
    lldbPluginProcessUtility
    LLVMTestingSupport)

add_lldb_unittest(RegisterSupportTests
  PARTIAL_SOURCES_INTENDED
  RegisterInfo_x86Test.cpp
  ${PLATFORM_SOURCES}

  LINK_LIBS
    lldbRegister
    lldbProcessContextSupport)
