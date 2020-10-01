set(SWIFT_SOURCES
  TestSwiftASTContext.cpp
  TestTypeSystemSwiftTypeRef.cpp
)
set(SWIFT_LIBS
  lldbPluginTypeSystemSwift
)
set(LLVM_OPTIONAL_SOURCES ${SWIFT_SOURCES})
if (NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_SOURCES)
  unset(SWIFT_LIBS)
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
    ${SWIFT_LIBS}
    LLVMTestingSupport
  )

set(test_inputs
  inlined-functions.yaml
  )
add_unittest_inputs(SymbolTests "${test_inputs}")
