add_lldb_library(lldbPluginProcessAIX
  NativeProcessAIX.cpp
  NativeThreadAIX.cpp
  NativeRegisterContextAIX.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessPOSIX
    lldbPluginProcessUtility
  )

target_compile_definitions(lldbPluginProcessAIX PRIVATE "-D_ALL_SOURCE")
