set(SWIFT_SOURCES HostInfoSwiftTest.cpp)
set(LLVM_OPTIONAL_SOURCES ${SWIFT_SOURCES})
if (NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_SOURCES)
endif()

set (FILES
  ConnectionFileDescriptorTest.cpp
  FileActionTest.cpp
  FileSystemTest.cpp
  FileTest.cpp
  HostInfoTest.cpp
  HostTest.cpp
  MainLoopTest.cpp
  NativeProcessProtocolTest.cpp
  PipeTest.cpp
  ProcessLaunchInfoTest.cpp
  SocketAddressTest.cpp
  SocketTest.cpp
  ThreadLauncherTest.cpp
  XMLTest.cpp
  ${SWIFT_SOURCES}
)

if (CMAKE_SYSTEM_NAME MATCHES "Linux|Android")
  list(APPEND FILES
    linux/HostTest.cpp
    linux/SupportTest.cpp
  )
endif()

if (LLDB_ENABLE_TERMIOS)
  list(APPEND FILES
    posix/TerminalTest.cpp
  )
endif()

add_lldb_unittest(HostTests
  ${FILES}
  LINK_LIBS
    lldbHost
    lldbCore
    lldbUtilityHelpers
    lldbHostHelpers
    LLVMTestingSupport
  )
