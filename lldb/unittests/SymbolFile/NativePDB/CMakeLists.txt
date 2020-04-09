add_lldb_unittest(SymbolFileNativePDBTests
  PdbFPOProgramToDWARFExpressionTests.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbPluginSymbolFileNativePDB
    lldbUtilityHelpers
    lldbPluginPlatformMacOSX
  LINK_COMPONENTS
    Support
    DebugInfoPDB
  )
