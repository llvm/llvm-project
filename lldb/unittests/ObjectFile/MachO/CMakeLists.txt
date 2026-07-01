add_lldb_unittest(ObjectFileMachOTests
  TestObjectFileMachO.cpp
  MachOTrieTest.cpp

  LINK_LIBS
    lldbPluginObjectFileMachO
    lldbPluginSymbolFileSymtab
    lldbCore
    lldbUtilityHelpers
    lldbPluginPlatformMacOSX
    LLVMTestingSupport
  )
