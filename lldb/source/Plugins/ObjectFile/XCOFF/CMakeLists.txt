add_lldb_library(lldbPluginObjectFileXCOFF PLUGIN
  ObjectFileXCOFF.cpp

  LINK_COMPONENTS
    BinaryFormat
    Object
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
  )
