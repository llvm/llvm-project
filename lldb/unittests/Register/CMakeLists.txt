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

add_lldb_unittest(RegisterTests
  DumpRegisterInfoTest.cpp
  RegisterContextTest.cpp
  RegisterDefinitionTest.cpp
  RegisterDescriptionFinalizationTest.cpp
  RegisterInfoARMTest.cpp
  RegisterInfoLoongArchTest.cpp
  RegisterInfoPowerPCTest.cpp
  RegisterInfoRISCVTest.cpp
  RegisterInfoS390XTest.cpp
  RegisterInfoX86LayoutTest.cpp
  RegisterInfoX86ModelTest.cpp
  RegisterInfo_x86Test.cpp
  RegisterLayoutBankTest.cpp
  RegisterLayoutTest.cpp
  RegisterTypeTest.cpp
  RegisterValueTest.cpp
  ${PLATFORM_SOURCES}

  LINK_LIBS
    lldbRegister
    lldbUtility
    LLVMTestingSupport)
