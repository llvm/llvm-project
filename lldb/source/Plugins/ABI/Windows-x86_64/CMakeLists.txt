add_lldb_library(lldbPluginABIWindows_x86_64 PLUGIN
  ABIWindows_x86_64.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
