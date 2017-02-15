add_lldb_unittest(UtilityTests
  ConstStringTest.cpp
  StringExtractorTest.cpp
  TaskPoolTest.cpp
  TimeoutTest.cpp
  UriParserTest.cpp

  LINK_LIBS
      lldbHost
      lldbUtility
    LINK_COMPONENTS
      Support
  )
