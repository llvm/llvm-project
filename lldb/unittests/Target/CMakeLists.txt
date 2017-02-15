add_lldb_unittest(TargetTests
  ModuleCacheTest.cpp

  LINK_LIBS
      lldbCore
      lldbHost
      lldbSymbol
      lldbUtility
      lldbPluginObjectFileELF
    LINK_COMPONENTS
      Support
  )

add_unittest_inputs(TargetTests TestModule.so)
