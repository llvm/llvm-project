add_lldb_library(lldbPluginProcessWasm PLUGIN
  ProcessWasm.cpp
  RegisterContextWasm.cpp
  ThreadWasm.cpp
  UnwindWasm.cpp

  LINK_LIBS
    lldbCore
    lldbRegister
  LINK_COMPONENTS
    Support
  )
