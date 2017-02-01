add_lldb_library(lldbPluginABISysV_hexagon PLUGIN
  ABISysV_hexagon.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
