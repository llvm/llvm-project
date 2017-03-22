add_subdirectory(Mocks)

add_lldb_unittest(UtilityTests
  ConstStringTest.cpp
  ErrorTest.cpp
  LogTest.cpp
  NameMatchesTest.cpp
  StringExtractorTest.cpp
  TaskPoolTest.cpp
  TildeExpressionResolverTest.cpp
  TimeoutTest.cpp
  UriParserTest.cpp
  VASprintfTest.cpp

  LINK_LIBS
      lldbUtility
      lldbUtilityMocks
  LINK_COMPONENTS
    Support
  )
