add_lldb_unittest(LLDBValueObjectTests
  DumpValueObjectOptionsTests.cpp

  LINK_LIBS
    lldbPluginPlatformLinux
    lldbPluginScriptInterpreterNone

  LINK_COMPONENTS
    Support
  )
