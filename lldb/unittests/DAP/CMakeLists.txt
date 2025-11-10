add_lldb_unittest(DAPTests
  ClientLauncherTest.cpp
  DAPErrorTest.cpp
  DAPTest.cpp
  DAPTypesTest.cpp
  FifoFilesTest.cpp
  Handler/DisconnectTest.cpp
  Handler/ContinueTest.cpp
  JSONUtilsTest.cpp
  LLDBUtilsTest.cpp
  ProtocolRequestsTest.cpp
  ProtocolTypesTest.cpp
  ProtocolUtilsTest.cpp
  TestBase.cpp
  VariablesTest.cpp

  SBAPITEST

  LINK_COMPONENTS
    Support
  LINK_LIBS
    liblldb
    lldbDAP
    lldbUtilityHelpers
    LLVMTestingSupport
  )

set(test_inputs
  linux-x86_64.out.yaml
  linux-x86_64.core.yaml
  )
add_unittest_inputs(DAPTests "${test_inputs}")
