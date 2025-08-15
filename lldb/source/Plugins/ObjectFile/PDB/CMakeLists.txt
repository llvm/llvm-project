add_lldb_library(lldbPluginObjectFilePDB PLUGIN
  ObjectFilePDB.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbUtility
  )
