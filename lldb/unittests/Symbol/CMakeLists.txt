set(LLVM_OPTIONAL_SOURCES TestTypeSystemFortran.cpp)

set(FORTRAN_TESTS "")
if(LLDB_ENABLE_FORTRAN)
  set(FORTRAN_TESTS TestTypeSystemFortran.cpp)
endif()

add_lldb_unittest(SymbolTests
  JSONSymbolTest.cpp
  LineTableTest.cpp
  LocateSymbolFileTest.cpp
  MangledTest.cpp
  PostfixExpressionTest.cpp
  SymbolTest.cpp
  SymtabTest.cpp
  SymStoreTest.cpp
  TestTypeSystem.cpp
  TestTypeSystemClang.cpp
  TestClangASTImporter.cpp
  TestDWARFCallFrameInfo.cpp
  TestType.cpp
  TestLineEntry.cpp
  UnwindPlanTest.cpp
  ${FORTRAN_TESTS}

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbUtilityHelpers
    lldbPluginObjectFileELF
    lldbPluginObjectFileMachO
    lldbPluginSymbolFileDWARF
    lldbPluginSymbolFileSymtab
    lldbPluginTypeSystemClang
    lldbPluginSymbolLocatorSymStore
    LLVMTestingSupport
  )

if(LLDB_ENABLE_FORTRAN)
  target_link_libraries(SymbolTests PRIVATE 
    lldbPluginTypeSystemFortran
  )
endif()


set(test_inputs
  inlined-functions.yaml
  )
add_unittest_inputs(SymbolTests "${test_inputs}")
