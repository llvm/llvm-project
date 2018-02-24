add_lldb_unittest(UtilityTests
  ArchSpecTest.cpp
  CleanUpTest.cpp
  ConstStringTest.cpp
  EnvironmentTest.cpp
  JSONTest.cpp
  LogTest.cpp
  NameMatchesTest.cpp
  StatusTest.cpp
  StringExtractorTest.cpp
  StructuredDataTest.cpp
  TildeExpressionResolverTest.cpp
  TimeoutTest.cpp
  TimerTest.cpp
  UriParserTest.cpp
  VASprintfTest.cpp

  LINK_LIBS
      lldbUtility
      lldbUtilityHelpers
  LINK_COMPONENTS
    Support
  )

add_unittest_inputs(UtilityTests
  StructuredData-basic.json
  )
