add_lldb_unittest(TraceIntelPTTests
  IntelPTCollectorTests.cpp

  LINK_LIBS
    lldbPluginProcessLinux
  )

target_include_directories(TraceIntelPTTests PRIVATE
  ${LLDB_SOURCE_DIR}/source/Plugins/Process/Linux)
