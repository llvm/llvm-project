add_lldb_unittest(ScriptInterpreterLuaTests
  LuaTests.cpp
  ScriptInterpreterTests.cpp

  LINK_LIBS
    lldbHost
    lldbPluginScriptInterpreterLua
    lldbPluginPlatformLinux
    LLVMTestingSupport
  LINK_COMPONENTS
    Support
  )
