add_lldb_unittest(LLDBCoreTests
  ArchSpecTest.cpp
  BroadcasterTest.cpp
  DataExtractorTest.cpp
  ErrorTest.cpp
  ListenerTest.cpp
  LogTest.cpp
  ScalarTest.cpp
  StateTest.cpp
  StructuredDataTest.cpp
  TimerTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
  LINK_COMPONENTS
    Support
  )
