add_lldb_unittest(ObjectFileBreakpadTests
  BreakpadRecordsTest.cpp

  LINK_LIBS
    lldbPluginObjectFileBreakpad
  )
