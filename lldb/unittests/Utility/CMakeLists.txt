add_lldb_unittest(UtilityTests
  ConstStringTest.cpp
  ModuleCacheTest.cpp
  StringExtractorTest.cpp
  TaskPoolTest.cpp
  TimeoutTest.cpp
  UriParserTest.cpp

  LINK_LIBS
      lldbCore
      lldbHost
      lldbSymbol
      lldbUtility
      lldbPluginObjectFileELF
    LINK_COMPONENTS
      Support
  )

add_unittest_inputs(UtilityTests TestModule.so)
