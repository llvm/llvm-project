add_lldb_library(lldbPluginABISysV_s390x PLUGIN
  ABISysV_s390x.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
