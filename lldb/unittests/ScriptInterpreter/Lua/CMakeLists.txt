add_lldb_unittest(ScriptInterpreterLuaTests
  LuaTests.cpp
  ScriptInterpreterTests.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbHost
    lldbPluginScriptInterpreterLua
    lldbPluginPlatformLinux
    LLVMTestingSupport
  )
