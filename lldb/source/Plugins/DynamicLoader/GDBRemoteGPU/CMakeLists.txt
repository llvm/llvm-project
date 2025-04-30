add_lldb_library(lldbPluginDynamicLoaderGDBRemoteGPU PLUGIN
  DynamicLoaderGDBRemoteGPU.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
  )
