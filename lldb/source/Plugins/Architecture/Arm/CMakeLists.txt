add_lldb_library(lldbPluginArchitectureArm PLUGIN
  ArchitectureArm.cpp

  LINK_LIBS
    lldbPluginProcessUtility
    lldbCore
    lldbTarget
  LINK_COMPONENTS
    Support
  )
