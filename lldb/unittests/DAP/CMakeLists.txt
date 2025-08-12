add_lldb_unittest(DAPTests
  DAPErrorTest.cpp
  DAPTest.cpp
  FifoFilesTest.cpp
  Handler/DisconnectTest.cpp
  Handler/ContinueTest.cpp
  JSONUtilsTest.cpp
  LLDBUtilsTest.cpp
  ProtocolTypesTest.cpp
  ProtocolUtilsTest.cpp
  TestBase.cpp
  VariablesTest.cpp

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
