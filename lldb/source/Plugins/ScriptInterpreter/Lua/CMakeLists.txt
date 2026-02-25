add_lldb_library(lldbPluginScriptInterpreterLua PLUGIN
  LuaState.cpp
  ScriptInterpreterLua.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
  )

target_include_directories(lldbPluginScriptInterpreterLua PUBLIC ${LUA_INCLUDE_DIR})
target_link_libraries(lldbPluginScriptInterpreterLua INTERFACE ${LUA_LIBRARIES})
