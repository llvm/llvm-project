if (LLDB_ENABLE_DYNAMIC_SCRIPTINTERPRETERS)
  set(lua_plugin_lib lldbStaticScriptInterpreterLua)
else()
  set(lua_plugin_lib lldbPluginScriptInterpreterLua)
endif()

add_lldb_unittest(ScriptInterpreterLuaTests
  LuaTests.cpp
  ScriptInterpreterTests.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbHost
    ${lua_plugin_lib}
    lldbPluginPlatformLinux
    LLVMTestingSupport
  )
