add_lldb_unittest(SymbolTests
  LocateSymbolFileTest.cpp
  TestClangASTContext.cpp
  TestDWARFCallFrameInfo.cpp
  TestType.cpp
  TestSwiftASTContext.cpp

  LINK_LIBS
    lldbHost
    lldbSymbol
    lldbUtilityHelpers
    lldbPluginObjectFileELF
  )

add_dependencies(SymbolTests yaml2obj)
add_definitions(-DYAML2OBJ="$<TARGET_FILE:yaml2obj>")
set(test_inputs
  basic-call-frame-info.yaml
  )
add_unittest_inputs(SymbolTests "${test_inputs}")
