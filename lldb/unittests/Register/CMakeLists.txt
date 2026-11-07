add_lldb_unittest(RegisterTests
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
