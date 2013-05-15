include_directories(../Utility)

set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginProcessMachCore
  ProcessMachCore.cpp
  ThreadMachCore.cpp
  )
