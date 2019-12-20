add_lldb_library(lldbPluginScriptInterpreterLua PLUGIN
  ScriptInterpreterLua.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
  )