add_lldb_library(lldbPluginArchitectureArm PLUGIN
  ArchitectureArm.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbPluginProcessUtility
    lldbCore
    lldbTarget
    lldbUtility
  )
