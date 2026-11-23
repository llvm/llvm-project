add_lldb_library(lldbPluginPlatformAIX PLUGIN
  PlatformAIX.cpp

   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
    lldbPluginPlatformPOSIX
  )

target_compile_definitions(lldbPluginPlatformAIX PRIVATE "-D_ALL_SOURCE")
