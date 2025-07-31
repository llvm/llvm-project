add_lldb_library(lldbPluginProcessWasm PLUGIN
  ProcessWasm.cpp
  RegisterContextWasm.cpp
  ThreadWasm.cpp
  UnwindWasm.cpp

  LINK_LIBS
    lldbCore
  LINK_COMPONENTS
    Support
  )
