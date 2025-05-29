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
    liblldb
    lldbDAP
    lldbUtilityHelpers
    LLVMTestingSupport
  LINK_COMPONENTS
    Support
  )

set(test_inputs
  linux-x86_64.out.yaml
  linux-x86_64.core.yaml
  )
add_unittest_inputs(DAPTests "${test_inputs}")
