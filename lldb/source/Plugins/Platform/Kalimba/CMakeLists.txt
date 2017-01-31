add_lldb_library(lldbPluginPlatformKalimba PLUGIN
  PlatformKalimba.cpp

   LINK_LIBS
    lldbCore
    lldbHost
    lldbTarget
  )
