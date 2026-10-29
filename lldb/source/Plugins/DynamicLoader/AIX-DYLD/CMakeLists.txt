add_definitions("-D_ALL_SOURCE")

add_lldb_library(lldbPluginDynamicLoaderAIXDYLD PLUGIN
  DynamicLoaderAIXDYLD.cpp

  LINK_LIBS
    lldbCore
    lldbTarget
  LINK_COMPONENTS
    Support
  )
