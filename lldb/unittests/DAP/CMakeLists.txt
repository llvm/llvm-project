add_lldb_unittest(DAPTests
  DAPTest.cpp
  Handler/DisconnectRequestHandlerTest.cpp
  JSONUtilsTest.cpp
  LLDBUtilsTest.cpp
  ProtocolTypesTest.cpp
  TestBase.cpp
  TransportTest.cpp

  LINK_LIBS
    lldbDAP
    LLVMTestingSupport
  LINK_COMPONENTS
    Support
  )
