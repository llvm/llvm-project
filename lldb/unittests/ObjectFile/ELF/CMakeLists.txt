add_lldb_unittest(ObjectFileELFTests
  TestObjectFileELF.cpp

  LINK_LIBS
    lldbPluginObjectFileELF
    lldbPluginSymbolFileSymtab
    lldbCore
    lldbUtilityHelpers
    LLVMTestingSupport
  )

set(test_inputs
  early-section-headers.so
  )
add_unittest_inputs(ObjectFileELFTests "${test_inputs}")
