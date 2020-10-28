add_lldb_unittest(ProcessorTraceTests
  ProcessorTraceTest.cpp

  LINK_LIBS
    lldbPluginProcessLinux
  )

target_include_directories(ProcessorTraceTests PRIVATE
  ${LLDB_SOURCE_DIR}/source/Plugins/Process/Linux)
