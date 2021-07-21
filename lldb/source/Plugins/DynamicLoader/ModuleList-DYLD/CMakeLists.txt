add_lldb_library(lldbPluginDynamicLoaderDumpWithModuleList PLUGIN
  DynamicLoaderDumpWithModuleList.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbTarget
    lldbPluginDynamicLoaderPosixDYLD
    lldbPluginProcessElfCore
  LINK_COMPONENTS
    Support
  )
