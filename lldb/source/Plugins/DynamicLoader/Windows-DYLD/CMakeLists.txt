add_lldb_library(lldbPluginDynamicLoaderWindowsDYLD PLUGIN
  DynamicLoaderWindowsDYLD.cpp

  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbCore
    lldbTarget
  )
