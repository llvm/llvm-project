add_lldb_unittest(ObjectFileMachOTests
  TestObjectFileMachO.cpp

  LINK_LIBS
    lldbPluginObjectFileMachO
    lldbPluginSymbolFileSymtab
    lldbCore
    lldbUtilityHelpers
    LLVMTestingSupport
  )
