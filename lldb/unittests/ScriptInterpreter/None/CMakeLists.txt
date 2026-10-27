add_lldb_unittest(ScriptInterpreterNoneTests
  ScriptInterpreterNoneTest.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginPlatformLinux
    lldbPluginScriptInterpreterNone
    LLVMTestingSupport
  )
