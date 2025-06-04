add_lldb_library(lldbPluginArchitecturePPC64 PLUGIN
  ArchitecturePPC64.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbPluginProcessUtility
    lldbCore
    lldbTarget
    lldbUtility
  )
