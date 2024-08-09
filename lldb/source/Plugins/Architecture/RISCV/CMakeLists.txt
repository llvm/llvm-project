add_lldb_library(lldbPluginArchitectureRISCV PLUGIN
  ArchitectureRISCV.cpp
  DirectToIndirectFCR.cpp

  LINK_LIBS
    lldbPluginProcessUtility
    lldbCore
    lldbTarget
    lldbUtility
  LINK_COMPONENTS
    Support
  )
