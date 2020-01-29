add_lldb_unittest(SymbolFilePDBTests
  SymbolFilePDBTests.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbPluginObjectFilePECOFF
    lldbPluginSymbolFileDWARF
    lldbPluginSymbolFilePDB
    lldbPluginTypeSystemClang
    lldbUtilityHelpers
    LLVMTestingSupport
  LINK_COMPONENTS
    Support
    DebugInfoPDB
  )

set(test_inputs
   test-pdb.exe
   test-pdb.pdb
   test-pdb-types.exe
   test-pdb-types.pdb)

add_unittest_inputs(SymbolFilePDBTests "${test_inputs}")
