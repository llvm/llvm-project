add_lldb_library(lldbPluginScriptedFrameProvider PLUGIN
  ScriptedFrameProvider.cpp

  LINK_COMPONENTS
    Support

  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbTarget
    lldbUtility
  )
