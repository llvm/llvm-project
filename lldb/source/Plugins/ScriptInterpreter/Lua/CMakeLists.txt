set(lua_plugin_sources
  LuaState.cpp
  ScriptInterpreterLua.cpp
)

if (LLDB_ENABLE_DYNAMIC_SCRIPTINTERPRETERS)
  # Shared library loaded at runtime by PluginManager. Private lldb symbols are
  # resolved via liblldb's re-exports, so lldb_private libs aren't linked here.
  add_lldb_library(lldbPluginScriptInterpreterLua SHARED
    ${lua_plugin_sources}
    LINK_LIBS
      liblldb
      ${LUA_LIBRARIES}
  )
  add_lua_wrapper(lldbPluginScriptInterpreterLua)

  # Static variant linked directly by unit tests. Separate compilation is
  # required so llvm::Error RTTI (ErrorInfoBase::ID) has a single address
  # shared between the test binary and the plugin code.
  add_lldb_library(lldbStaticScriptInterpreterLua
    ${lua_plugin_sources}
    LINK_LIBS
      lldbCore
      lldbInterpreter
      ${LUA_LIBRARIES}
  )
  target_include_directories(lldbStaticScriptInterpreterLua
    PUBLIC ${LUA_INCLUDE_DIR})
else()
  add_lldb_library(lldbPluginScriptInterpreterLua PLUGIN
    ${lua_plugin_sources}
    LINK_LIBS
      lldbCore
      lldbInterpreter
      ${LUA_LIBRARIES}
  )
endif()

target_include_directories(lldbPluginScriptInterpreterLua
  PUBLIC ${LUA_INCLUDE_DIR})
