add_lldb_library(lldbPluginABISysV_i386 PLUGIN
  ABISysV_i386.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
