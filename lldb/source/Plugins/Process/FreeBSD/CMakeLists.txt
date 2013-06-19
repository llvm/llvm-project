set(LLVM_NO_RTTI 1)

include_directories(.)
include_directories(../POSIX)

add_lldb_library(lldbPluginProcessFreeBSD
  ProcessFreeBSD.cpp
  ProcessMonitor.cpp
  )
