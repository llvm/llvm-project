add_lldb_unittest(LLDBCoreTests
  CommunicationTest.cpp
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
