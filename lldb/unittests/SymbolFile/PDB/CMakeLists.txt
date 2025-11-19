add_lldb_unittest(SymbolFilePDBTests
  SymbolFilePDBTests.cpp

  LINK_COMPONENTS
    Support
    DebugInfoPDB
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbPluginObjectFilePECOFF
    lldbPluginPlatformWindows
    lldbPluginSymbolFileDWARF
    lldbPluginSymbolFilePDB
    lldbPluginTypeSystemClang
    lldbUtilityHelpers
    LLVMTestingSupport
  )

set(test_inputs
   test-pdb.exe
   test-pdb.pdb
   test-pdb-types.exe
   test-pdb-types.pdb)

add_unittest_inputs(SymbolFilePDBTests "${test_inputs}")
