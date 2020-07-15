set(SWIFT_SOURCES
  TestSwiftASTContext.cpp
  TestTypeSystemSwiftTypeRef.cpp
)
set(LLVM_OPTIONAL_SOURCES ${SWIFT_SOURCES})
if (NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_SOURCES)
endif()

add_lldb_unittest(SymbolTests
  LocateSymbolFileTest.cpp
  PostfixExpressionTest.cpp
  TestTypeSystemClang.cpp
  TestClangASTImporter.cpp
  TestDWARFCallFrameInfo.cpp
  TestType.cpp
  TestLineEntry.cpp

  ${SWIFT_SOURCES}

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
    LLVMTestingSupport
  )

set(test_inputs
  inlined-functions.yaml
  )
add_unittest_inputs(SymbolTests "${test_inputs}")
