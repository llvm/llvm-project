add_lldb_unittest(ProcessWasmTests
  RegisterContextWasmTest.cpp

  LINK_LIBS
    lldbPluginProcessWasm
  )
