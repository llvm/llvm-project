add_lldb_library(lldbPluginObjectFileXCOFF PLUGIN
  ObjectFileXCOFF.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    BinaryFormat
    Object
    Support
  )
