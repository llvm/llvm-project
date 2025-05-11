add_lldb_unittest(DAPTests
  JSONUtilsTest.cpp

  LINK_LIBS
    lldbDAP
  LINK_COMPONENTS
    Support
  )
