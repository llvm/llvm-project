add_lldb_unittest(ProcessUtilityTests
  RegisterContextTest.cpp
  RegisterContextFreeBSDTest.cpp
  LinuxProcMapsTest.cpp

  LINK_LIBS
    lldbPluginProcessUtility)
