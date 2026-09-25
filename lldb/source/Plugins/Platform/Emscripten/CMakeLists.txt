add_lldb_library(lldbPluginPlatformEmscripten PLUGIN
  PlatformEmscripten.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginPlatformPOSIX
    lldbTarget
  )
