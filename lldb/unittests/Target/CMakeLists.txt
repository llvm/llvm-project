add_lldb_unittest(TargetTests
  MemoryRegionInfoTest.cpp
  ModuleCacheTest.cpp
  PathMappingListTest.cpp
  ProcessInstanceInfoTest.cpp

  LINK_LIBS
      lldbCore
      lldbHost
      lldbSymbol
      lldbUtility
      lldbPluginObjectFileELF
      lldbPluginSymbolFileSymtab
      lldbUtilityHelpers
    LINK_COMPONENTS
      Support
  )

add_unittest_inputs(TargetTests TestModule.so)
