add_lldb_unittest(SymbolTests
  LocateSymbolFileTest.cpp
  PostfixExpressionTest.cpp
  TestTypeSystemClang.cpp
  TestClangASTImporter.cpp
  TestDWARFCallFrameInfo.cpp
  TestType.cpp
  TestTypeSystemSwiftTypeRef.cpp
  TestSwiftASTContext.cpp
  TestLineEntry.cpp

  LINK_LIBS
    lldbHost
    lldbSymbol
    lldbUtilityHelpers
    lldbPluginObjectFileELF
    lldbPluginObjectFileMachO
    lldbPluginSymbolFileDWARF
    lldbPluginSymbolFileSymtab
    lldbPluginTypeSystemClang
    lldbPluginTypeSystemSwift
    lldbPluginPlatformMacOSX
    LLVMTestingSupport
  )

set(test_inputs
  inlined-functions.yaml
  )
add_unittest_inputs(SymbolTests "${test_inputs}")
