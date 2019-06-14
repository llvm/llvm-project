add_lldb_unittest(LLDBCoreTests
  MangledTest.cpp
  RichManglingContextTest.cpp
  StreamCallbackTest.cpp
  UniqueCStringMapTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbPluginObjectFileELF
    lldbPluginSymbolVendorELF
    lldbUtilityHelpers
    LLVMTestingSupport
  LINK_COMPONENTS
    Support
  )

set(test_inputs
  mangled-function-names.yaml
  )
add_unittest_inputs(LLDBCoreTests "${test_inputs}")
