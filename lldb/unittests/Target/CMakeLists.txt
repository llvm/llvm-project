add_lldb_unittest(TargetTests
  ABITest.cpp
  ExecutionContextTest.cpp
  MemoryRegionInfoTest.cpp
  ModuleCacheTest.cpp
  PathMappingListTest.cpp
  RemoteAwarePlatformTest.cpp
  StackFrameRecognizerTest.cpp

  LINK_LIBS
      lldbCore
      lldbHost
      lldbPluginObjectFileELF
      lldbPluginPlatformLinux
      lldbPluginSymbolFileSymtab
      lldbTarget
      lldbSymbol
      lldbUtility
      lldbUtilityHelpers
    LINK_COMPONENTS
      Support
  )

add_unittest_inputs(TargetTests TestModule.so)
