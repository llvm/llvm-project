add_lldb_unittest(DAPTests
  JSONUtilsTest.cpp
  LLDBUtilsTest.cpp

  LINK_LIBS
    lldbDAP
  LINK_COMPONENTS
    Support
  )
