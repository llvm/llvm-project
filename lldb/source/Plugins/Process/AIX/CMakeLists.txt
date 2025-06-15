add_lldb_library(lldbPluginProcessAIX
  NativeProcessAIX.cpp
  NativeRegisterContextAIX.cpp
  NativeRegisterContextAIX_ppc64.cpp
  NativeThreadAIX.cpp

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
