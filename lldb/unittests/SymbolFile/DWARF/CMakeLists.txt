add_lldb_unittest(SymbolFileDWARFTests
  DWARFASTParserClangTests.cpp
  DWARFDebugNamesIndexTest.cpp
  DWARFDIETest.cpp
  DWARFIndexCachingTest.cpp
  DWARFUnitTest.cpp
  SymbolFileDWARFTests.cpp
  XcodeSDKModuleTests.cpp
  DWARF64UnitTest.cpp

  LINK_COMPONENTS
    Support
    DebugInfoPDB
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
    LLVMTestingSupport
  )

set(test_inputs
   test-dwarf.exe
   DW_AT_default_value-test.yaml)

add_unittest_inputs(SymbolFileDWARFTests "${test_inputs}")
