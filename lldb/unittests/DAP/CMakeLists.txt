add_lldb_unittest(DAPTests
  DAPTest.cpp
  FifoFilesTest.cpp
  Handler/DisconnectTest.cpp
  Handler/ContinueTest.cpp
  JSONUtilsTest.cpp
  LLDBUtilsTest.cpp
  ProtocolTypesTest.cpp
  TestBase.cpp
  TransportTest.cpp
  VariablesTest.cpp

  LINK_LIBS
    lldbDAP
    LLVMTestingSupport
  LINK_COMPONENTS
    Support
  )
