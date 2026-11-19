set(FREEBSD_SOURCES
  Layouts/RegisterLayoutFreeBSDNativeTest.cpp)
set(NETBSD_SOURCES
  Layouts/RegisterLayoutNetBSD_x86_32Test.cpp
  Layouts/RegisterLayoutNetBSD_x86_64Test.cpp)

if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  list(APPEND PLATFORM_SOURCES ${FREEBSD_SOURCES})
elseif (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
  list(APPEND PLATFORM_SOURCES ${NETBSD_SOURCES})
endif()

set(LLVM_OPTIONAL_SOURCES
  ${FREEBSD_SOURCES}
  ${NETBSD_SOURCES})

add_lldb_unittest(RegisterTests
  Descriptions/DumpRegisterDescriptionTest.cpp
  Descriptions/RegisterDescriptionTest.cpp
  Descriptions/RegisterDescription_arm32Test.cpp
  Descriptions/RegisterDescription_arm64Test.cpp
  Descriptions/RegisterDescription_ppcTest.cpp
  Descriptions/RegisterDescription_riscvTest.cpp
  Descriptions/RegisterDescription_x86Test.cpp
  Layouts/RegisterLayoutDarwinTest.cpp
  Layouts/RegisterLayoutFreeBSD_arm32Test.cpp
  Layouts/RegisterLayoutLinux_s390xTest.cpp
  Layouts/RegisterLayoutWindows_x86Test.cpp
  Layouts/RegisterLayout_arm64Test.cpp
  Layouts/RegisterLayout_loongarch64Test.cpp
  Layouts/RegisterLayout_ppcTest.cpp
  Layouts/RegisterLayout_riscv64Test.cpp
  Layouts/RegisterLayout_x86TagWordTest.cpp
  Layouts/RegisterLayout_x86Test.cpp
  RegisterContextTest.cpp
  RegisterDefinitionTest.cpp
  RegisterLayoutBankTest.cpp
  RegisterLayoutTest.cpp
  RegisterTypeTest.cpp
  RegisterValueTest.cpp
  ${PLATFORM_SOURCES}

  LINK_LIBS
    lldbRegister
    lldbUtility
    LLVMTestingSupport)
