add_lldb_library(lldbPluginABIMips PLUGIN
  ABIMips.cpp
  ABISysV_mips.cpp
  ABISysV_mips64.cpp

  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbValueObject
  )
