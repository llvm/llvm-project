add_lldb_library(lldbPluginABIHexagon PLUGIN
  ABISysV_hexagon.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbValueObject
  LINK_COMPONENTS
    Support
    TargetParser
  )
