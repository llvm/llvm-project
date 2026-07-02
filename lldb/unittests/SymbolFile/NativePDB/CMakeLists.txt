add_lldb_unittest(SymbolFileNativePDBTests
  PdbFPOProgramToDWARFExpressionTests.cpp
  SymbolFilePDBTests.cpp
  UdtRecordCompleterTests.cpp

  LINK_COMPONENTS
    Support
    DebugInfoPDB
    TestingSupport
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbPluginObjectFilePECOFF
    lldbPluginPlatformWindows
    lldbPluginSymbolFileNativePDB
    lldbPluginSymbolFilePDB # for the pdb reader setting
    lldbUtilityHelpers
  )

set(test_inputs
   dynamic-types-exe.yaml
   dynamic-types.yaml)

add_unittest_inputs(SymbolFileNativePDBTests "${test_inputs}")
