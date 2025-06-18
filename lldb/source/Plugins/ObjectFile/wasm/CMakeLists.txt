add_lldb_library(lldbPluginObjectFileWasm PLUGIN
  ObjectFileWasm.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbUtility
  )
