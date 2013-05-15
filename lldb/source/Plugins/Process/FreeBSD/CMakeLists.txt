set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginProcessFreeBSD
  ProcessFreeBSD.cpp
  ProcessMonitor.cpp
  )
