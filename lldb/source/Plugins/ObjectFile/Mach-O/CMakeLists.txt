add_lldb_library(lldbPluginObjectFileMachO PLUGIN
  ObjectFileMachO.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessUtility
  )
