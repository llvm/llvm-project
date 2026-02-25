add_lldb_library(lldbPluginScriptInterpreterLua PLUGIN
  LuaState.cpp
  ScriptInterpreterLua.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
    ${LUA_LIBRARIES}
  )

  target_include_directories(lldbPluginScriptInterpreterLua
    PUBLIC ${LUA_INCLUDE_DIR})
