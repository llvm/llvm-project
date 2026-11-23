add_lldb_library(lldbPluginObjectContainerClangOffloadBundle PLUGIN
  ObjectContainerClangOffloadBundle.cpp

  LINK_COMPONENTS
    Object
    BinaryFormat
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
  )
