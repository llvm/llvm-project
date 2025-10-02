add_lldb_library(lldbPluginObjectFileELF PLUGIN
  ELFHeader.cpp
  ObjectFileELF.cpp

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
