add_lldb_unittest(ObjectFileELFTests
  TestELFHeader.cpp
  TestObjectFileELF.cpp

  LINK_LIBS
    lldbPluginObjectFileELF
    lldbPluginSymbolVendorELF
    lldbCore
    lldbUtilityHelpers
  )

add_dependencies(ObjectFileELFTests yaml2obj)
add_definitions(-DYAML2OBJ="$<TARGET_FILE:yaml2obj>")

set(test_inputs
  sections-resolve-consistently.yaml
  )
add_unittest_inputs(ObjectFileELFTests "${test_inputs}")
