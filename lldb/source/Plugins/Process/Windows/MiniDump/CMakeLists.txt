include_directories(../../Utility)
include_directories(../Common)

add_lldb_library(lldbPluginProcessWinMiniDump
  ProcessWinMiniDump.cpp
  RegisterContextWindowsMiniDump.cpp
  ThreadWinMiniDump.cpp
  )
