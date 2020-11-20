add_lldb_unittest(ProcessUtilityTests
  RegisterContextFreeBSDTest.cpp
  LinuxProcMapsTest.cpp

  LINK_LIBS
    lldbPluginProcessUtility)
