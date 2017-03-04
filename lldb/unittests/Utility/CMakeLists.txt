add_lldb_unittest(UtilityTests
  ConstStringTest.cpp
  ErrorTest.cpp
  LogTest.cpp
  NameMatchesTest.cpp
  StringExtractorTest.cpp
  TaskPoolTest.cpp
  TimeoutTest.cpp
  UriParserTest.cpp
  VASprintfTest.cpp

  LINK_LIBS
      lldbUtility
  LINK_COMPONENTS
    Support
  )
