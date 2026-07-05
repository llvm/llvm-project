add_lldb_library(lldbPluginDynamicLoaderAcceleratorGDBRemote PLUGIN
  DynamicLoaderAcceleratorGDBRemote.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessGDBRemote
  LINK_COMPONENTS
    Support
  )
