include_directories(${LLDB_SOURCE_DIR}/source/Plugins/Process/Linux)

add_lldb_unittest(ProcessorTraceTests
  ProcessorTraceTest.cpp

  LINK_LIBS
    lldbPluginProcessLinux
  )