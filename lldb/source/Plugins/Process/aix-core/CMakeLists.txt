add_definitions("-D_ALL_SOURCE")

add_lldb_library(lldbPluginProcessAIXCore PLUGIN
  ProcessAIXCore.cpp
  AIXCore.cpp
  ThreadAIXCore.cpp
  RegisterContextCoreAIX_ppc64.cpp

  LINK_LIBS
    lldbCore
    lldbTarget
    lldbPluginProcessUtility
  LINK_COMPONENTS
    BinaryFormat
    Support
  )
