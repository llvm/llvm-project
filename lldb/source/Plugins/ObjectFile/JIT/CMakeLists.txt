add_lldb_library(lldbPluginObjectFileJIT PLUGIN
  ObjectFileJIT.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
