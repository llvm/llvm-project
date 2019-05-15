add_lldb_unittest(SymbolTests
  LocateSymbolFileTest.cpp
  PostfixExpressionTest.cpp
  TestClangASTContext.cpp
  TestDWARFCallFrameInfo.cpp
  TestType.cpp
  TestLineEntry.cpp

  LINK_LIBS
    lldbHost
    lldbSymbol
    lldbUtilityHelpers
    lldbPluginObjectFileELF
    lldbPluginObjectFileMachO
    lldbPluginSymbolFileDWARF
    LLVMTestingSupport
  )

set(test_inputs
  basic-call-frame-info.yaml
  inlined-functions.yaml
  )
add_unittest_inputs(SymbolTests "${test_inputs}")
