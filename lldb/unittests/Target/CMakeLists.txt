add_lldb_unittest(TargetTests
  MemoryRegionInfoTest.cpp
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
