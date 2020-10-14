add_lldb_library(lldbPluginProcessTrace PLUGIN
  ProcessTrace.cpp

  LINK_LIBS
    lldbCore
    lldbTarget
    lldbUtility
    lldbPluginProcessUtility
  LINK_COMPONENTS
    BinaryFormat
    Object
    Support
  )
