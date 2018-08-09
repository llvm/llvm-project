add_lldb_unittest(LLDBCoreTests
  BroadcasterTest.cpp
  DataExtractorTest.cpp
  EventTest.cpp
  ListenerTest.cpp
  MangledTest.cpp
  ScalarTest.cpp
  StateTest.cpp
  StreamCallbackTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    LLVMTestingSupport
  LINK_COMPONENTS
    Support
  )
