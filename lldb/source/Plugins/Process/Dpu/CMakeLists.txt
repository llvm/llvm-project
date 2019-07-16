include_directories(${UPMEM_API_HEADERS})

add_lldb_library(lldbPluginProcessDpu PLUGIN
  DpuRank.cpp
  ProcessDpu.cpp
  RegisterContextDpu.cpp
  RegisterInfo_dpu.cpp
  ThreadDpu.cpp

  LINK_LIBS
    lldbCore
    lldbHostDpu
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessPOSIX
    lldbPluginProcessUtility
  LINK_COMPONENTS
    Support
  EXTRA_CXXFLAGS -g
  )
