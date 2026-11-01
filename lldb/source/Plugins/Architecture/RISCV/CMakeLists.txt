add_lldb_library(lldbPluginArchitectureRISCV PLUGIN
  ArchitectureRISCV.cpp

  LINK_LIBS
    lldbCore
    lldbUtility
  )
