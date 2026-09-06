add_lldb_library(lldbPluginObjectFileMachO PLUGIN
  ObjectFileMachO.cpp
  MachOTrie.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbProcessContextSupport
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
  )
