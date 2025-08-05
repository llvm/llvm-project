add_lldb_library(lldbPluginDynamicLoaderDumpWithModuleList PLUGIN
  DynamicLoaderDumpWithModuleList.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbTarget
    lldbPluginProcessElfCore
  LINK_COMPONENTS
    Support
  )
