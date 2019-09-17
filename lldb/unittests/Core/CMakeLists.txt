add_lldb_unittest(LLDBCoreTests
  MangledTest.cpp
  RichManglingContextTest.cpp
  StreamCallbackTest.cpp
  UniqueCStringMapTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbPluginObjectFileELF
    lldbPluginSymbolFileSymtab
    lldbUtilityHelpers
    LLVMTestingSupport
  LINK_COMPONENTS
    Support
  )
