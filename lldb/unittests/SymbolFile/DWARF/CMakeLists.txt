add_lldb_unittest(SymbolFileDWARFTests
  SymbolFileDWARFTests.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbPluginObjectFilePECOFF
    lldbPluginSymbolFileDWARF
    lldbPluginSymbolFilePDB
  LINK_COMPONENTS
    Support
    DebugInfoPDB
  )

set(test_inputs
   test-dwarf.exe)

add_unittest_inputs(SymbolFileDWARFTests "${test_inputs}")
