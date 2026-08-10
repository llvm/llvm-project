add_lldb_library(lldbPluginArchitectureAArch64 PLUGIN
  ArchitectureAArch64.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbPluginProcessUtility
    lldbCore
    lldbTarget
    lldbUtility
  )
