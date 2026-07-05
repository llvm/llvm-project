add_lldb_unittest(ProcessPOSIXTests
  NativeProcessELFTest.cpp

  LINK_LIBS
    lldbPluginProcessPOSIX
  )

target_include_directories(ProcessPOSIXTests PRIVATE
  ${LLDB_SOURCE_DIR}/source/Plugins/Process/POSIX)
