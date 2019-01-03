add_lldb_unittest(LLDBCoreTests
  MangledTest.cpp
  RangeTest.cpp
  RichManglingContextTest.cpp
  StreamCallbackTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbPluginObjectFileELF
    lldbPluginSymbolVendorELF
    lldbUtilityHelpers
  LINK_COMPONENTS
    Support
  )

add_dependencies(LLDBCoreTests yaml2obj)
add_definitions(-DYAML2OBJ="$<TARGET_FILE:yaml2obj>")

set(test_inputs
  mangled-function-names.yaml
  )
add_unittest_inputs(LLDBCoreTests "${test_inputs}")
