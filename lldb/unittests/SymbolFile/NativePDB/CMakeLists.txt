add_lldb_unittest(SymbolFileNativePDBTests
  PdbFPOProgramToDWARFExpressionTests.cpp
  UdtRecordCompleterTests.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbPluginSymbolFileNativePDB
    lldbUtilityHelpers
  LINK_COMPONENTS
    Support
    DebugInfoPDB
  )
