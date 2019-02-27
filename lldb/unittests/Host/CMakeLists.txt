set (FILES
  FileActionTest.cpp
  FileSystemTest.cpp
  HostInfoTest.cpp
  HostTest.cpp
  MainLoopTest.cpp
  NativeProcessProtocolTest.cpp
  ProcessInfoTest.cpp
  ProcessLaunchInfoTest.cpp
  SocketAddressTest.cpp
  SocketTest.cpp
  TaskPoolTest.cpp
)

if (CMAKE_SYSTEM_NAME MATCHES "Linux|Android")
  list(APPEND FILES
    linux/HostTest.cpp
    linux/SupportTest.cpp
  )
endif()

add_lldb_unittest(HostTests
  ${FILES}
  LINK_LIBS
    lldbCore
    lldbHost
    lldbUtilityHelpers
    LLVMTestingSupport
  )
