add_lldb_library(lldbPluginObjectContainerClangOffloadBundle PLUGIN
  ObjectContainerClangOffloadBundle.cpp

  LINK_COMPONENTS
    BinaryFormat
    Object
    TargetParser
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
  )
