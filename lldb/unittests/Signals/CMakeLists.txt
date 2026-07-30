add_lldb_unittest(SignalsTests
  UnixSignalsTest.cpp

  LINK_LIBS
    lldbTarget
  )
