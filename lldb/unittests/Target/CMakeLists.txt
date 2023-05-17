add_lldb_unittest(TargetTests
  ABITest.cpp
  DynamicRegisterInfoTest.cpp
  ExecutionContextTest.cpp
  MemoryRegionInfoTest.cpp
  MemoryTest.cpp
  MemoryTagMapTest.cpp
  ModuleCacheTest.cpp
  PathMappingListTest.cpp
  RegisterFlagsTest.cpp
  RemoteAwarePlatformTest.cpp
  StackFrameRecognizerTest.cpp
  FindFileTest.cpp

  LINK_LIBS
      lldbCore
      lldbHost
      lldbPluginObjectFileELF
      lldbPluginPlatformLinux
      lldbPluginPlatformMacOSX
      lldbPluginSymbolFileSymtab
      lldbTarget
      lldbSymbol
      lldbUtility
      lldbUtilityHelpers
    LINK_COMPONENTS
      Support
  )

add_unittest_inputs(TargetTests TestModule.so)
