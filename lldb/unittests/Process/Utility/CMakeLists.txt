add_lldb_unittest(ProcessUtilityTests
  LinuxProcMapsTest.cpp
  MemoryTagManagerAArch64MTETest.cpp
  CoreFileMemoryRangesTest.cpp

  LINK_LIBS
    lldbPluginProcessUtility
    LLVMTestingSupport)
