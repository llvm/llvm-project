add_lldb_unittest(RegisterTests
  Descriptions/DumpRegisterDescriptionTest.cpp
  Descriptions/RegisterDescriptionTest.cpp
  RegisterContextTest.cpp
  RegisterLayoutBankTest.cpp
  RegisterLayoutTest.cpp
  RegisterTypeTest.cpp
  RegisterValueTest.cpp

  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbRegister
    lldbUtilityHelpers
    LLVMTestingSupport
  )
