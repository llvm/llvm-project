add_lldb_unittest(ProcessUtilityTests
  RegisterContextTest.cpp
  RegisterContextFreeBSDTest.cpp

  LINK_LIBS
    lldbPluginProcessUtility)
