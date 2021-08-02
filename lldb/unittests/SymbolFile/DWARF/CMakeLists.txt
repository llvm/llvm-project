add_lldb_unittest(SymbolFileDWARFTests
  DWARFASTParserClangTests.cpp
  DWARFDIETest.cpp
  DWARFUnitTest.cpp
  SymbolFileDWARFTests.cpp
  XcodeSDKModuleTests.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbPluginObjectFilePECOFF
    lldbPluginSymbolFileDWARF
    lldbPluginSymbolFilePDB
    lldbPluginTypeSystemClang
    lldbPluginPlatformMacOSX
    lldbUtilityHelpers
    lldbSymbolHelpers
  LINK_COMPONENTS
    Support
    DebugInfoPDB
  )

set(test_inputs
   test-dwarf.exe)

add_unittest_inputs(SymbolFileDWARFTests "${test_inputs}")
