add_lldb_unittest(SymbolFileDWARFTests
  DWARFASTParserClangTests.cpp
  SymbolFileDWARFTests.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbPluginObjectFilePECOFF
    lldbPluginSymbolFileDWARF
    lldbPluginSymbolFilePDB
    lldbPluginTypeSystemClang
    lldbUtilityHelpers
  LINK_COMPONENTS
    Support
    DebugInfoPDB
  )

set(test_inputs
   test-dwarf.exe)

add_unittest_inputs(SymbolFileDWARFTests "${test_inputs}")
