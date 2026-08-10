add_lldb_unittest(ProcessLinuxTests
  PerfTests.cpp
  ProcfsTests.cpp

  LINK_LIBS
    lldbPluginProcessLinux
  )

target_include_directories(ProcessLinuxTests PRIVATE
  ${LLDB_SOURCE_DIR}/source/Plugins/Process/Linux)
