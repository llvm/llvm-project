add_lldb_library(lldbPluginObjectFilePECOFF PLUGIN
  ObjectFilePECOFF.cpp
  WindowsMiniDump.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    BinaryFormat
    Support
  )
