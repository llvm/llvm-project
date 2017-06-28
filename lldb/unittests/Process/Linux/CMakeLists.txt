include_directories(${LLDB_SOURCE_DIR}/source/Plugins/Process/Linux)

add_lldb_unittest(ProcessorTraceTest
  ProcessorTraceTest.cpp

  LINK_LIBS
    lldbPluginProcessLinux
  )