add_lldb_unittest(StackIDTests
  StackIDTest.cpp

  LINK_LIBS
      lldbCore
      lldbTarget
      lldbPluginPlatformMacOSX
  )
