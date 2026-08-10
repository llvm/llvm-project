add_lldb_library(lldbPluginObjectFileJSON PLUGIN
  ObjectFileJSON.cpp

  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbUtility
  )
