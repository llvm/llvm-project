include_directories(${LLDB_SOURCE_DIR}/source/Plugins/Process/POSIX)

add_lldb_unittest(ProcessPOSIXTest
  NativeProcessELFTest.cpp

  LINK_LIBS
    lldbPluginProcessPOSIX
  )
