add_lldb_unittest(UtilityTests
  ModuleCacheTest.cpp
  StringExtractorTest.cpp
  TaskPoolTest.cpp
  TimeoutTest.cpp
  UriParserTest.cpp
  )

add_unittest_inputs(UtilityTests TestModule.so)
