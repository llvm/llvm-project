set (FILES
  ConnectionFileDescriptorTest.cpp
  FileActionTest.cpp
  FileSystemTest.cpp
  FileTest.cpp
  HostInfoTest.cpp
  HostTest.cpp
  MainLoopTest.cpp
  NativeProcessProtocolTest.cpp
  ProcessLaunchInfoTest.cpp
  SocketAddressTest.cpp
  SocketTest.cpp
  SocketTestUtilities.cpp
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
    lldbHost
    lldbUtilityHelpers
    LLVMTestingSupport
  )
