add_lldb_library(lldbPluginArchitectureMips PLUGIN
  ArchitectureMips.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
    lldbUtility
  )
