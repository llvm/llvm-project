add_lldb_library(lldbPluginProcessMachCore PLUGIN
  ProcessMachCore.cpp
  ThreadMachCore.cpp
  RegisterContextUnifiedCore.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginDynamicLoaderDarwinKernel
    lldbPluginDynamicLoaderMacOSXDYLD
    lldbPluginObjectFileMachO
  )
