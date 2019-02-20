add_lldb_unittest(ObjectFileELFTests
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
  early-section-headers.so
  debug-info-relocations.pcm.yaml
  sections-resolve-consistently.yaml
  )
add_unittest_inputs(ObjectFileELFTests "${test_inputs}")
