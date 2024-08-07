add_lldb_library(lldbPluginScriptedPlatform PLUGIN
  ScriptedPlatform.cpp

  LINK_LIBS
    lldbCore
    lldbTarget
    lldbUtility
)
