add_lldb_library(lldbPluginABIMacOSX_i386 PLUGIN
  ABIMacOSX_i386.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
