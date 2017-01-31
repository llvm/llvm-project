add_lldb_library(lldbPluginPlatformPOSIX PLUGIN
  PlatformPOSIX.cpp

   LINK_LIBS
    lldbCore
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbTarget
  )
