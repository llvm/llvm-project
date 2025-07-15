add_lldb_library(lldbPluginObjectContainerBSDArchive PLUGIN
  ObjectContainerBSDArchive.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
  )
