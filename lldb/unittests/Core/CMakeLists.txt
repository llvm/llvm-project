add_lldb_unittest(LLDBCoreTests
  ArchSpecTest.cpp
  BroadcasterTest.cpp
  DataExtractorTest.cpp
  ListenerTest.cpp
  ScalarTest.cpp
  StateTest.cpp
  StreamCallbackTest.cpp
  StructuredDataTest.cpp
  TimerTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
  LINK_COMPONENTS
    Support
  )
