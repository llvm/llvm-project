add_lldb_library(lldbPluginABISystemZ PLUGIN
  ABISysV_s390x.cpp

  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbCore
    lldbRegister
    lldbSymbol
    lldbTarget
    lldbValueObject
  )
