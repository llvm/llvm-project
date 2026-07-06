set (FILES
  ConnectionFileDescriptorTest.cpp
  FileActionTest.cpp
  FileSystemTest.cpp
  FileTest.cpp
  HostInfoTest.cpp
  HostTest.cpp
  MainLoopTest.cpp
  JSONTransportTest.cpp
  NativeProcessProtocolTest.cpp
  PipeTest.cpp
  ProcessLaunchInfoTest.cpp
  ProcessRunLockTest.cpp
  SocketAddressTest.cpp
  SocketTest.cpp
  StripConPTYSequencesTest.cpp
  ThreadLauncherTest.cpp
  XMLTest.cpp
)

if (LLDB_ENABLE_PYTHON)
  list(APPEND FILES
    PythonRuntimeLoaderTest.cpp
  )
endif()

if (UNIX)
  list(APPEND FILES
    posix/HostTest.cpp
    posix/SupportTest.cpp
  )
endif()

if (LLDB_ENABLE_TERMIOS)
  list(APPEND FILES
    posix/TerminalTest.cpp
  )
endif()

add_lldb_unittest(HostTests
  ${FILES}
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS
    lldbHost
    lldbCore
    lldbUtilityHelpers
    lldbHostHelpers
    LLVMTestingSupport

  LINK_COMPONENTS
    TargetParser
  )

add_subdirectory(common)
