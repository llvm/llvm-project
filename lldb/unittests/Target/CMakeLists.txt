add_lldb_unittest(TargetTests
  ExecutionContextTest.cpp
  MemoryRegionInfoTest.cpp
  ModuleCacheTest.cpp
  PathMappingListTest.cpp

  LINK_LIBS
      lldbCore
      lldbHost
      lldbPluginObjectFileELF
      lldbPluginPlatformLinux
      lldbPluginSymbolFileSymtab
      lldbSymbol
      lldbUtility
      lldbUtilityHelpers
    LINK_COMPONENTS
      Support
  )

add_unittest_inputs(TargetTests TestModule.so)
