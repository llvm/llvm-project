add_subdirectory(Mocks)

add_lldb_unittest(UtilityTests
  ConstStringTest.cpp
  LogTest.cpp
  NameMatchesTest.cpp
  StatusTest.cpp
  StringExtractorTest.cpp
  StructuredDataTest.cpp
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

add_unittest_inputs(UtilityTests
  StructuredData-basic.json
  )
