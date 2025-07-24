add_lldb_unittest(LLDBMinidumpTests
  MinidumpParserTest.cpp
  RegisterContextMinidumpTest.cpp

  LINK_COMPONENTS
    ObjectYAML
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbTarget
    lldbPluginProcessUtility
    lldbPluginProcessMinidump
    lldbUtilityHelpers
    LLVMTestingSupport
  )

set(test_inputs
   fizzbuzz_no_heap.dmp
   fizzbuzz_wow64.dmp
   linux-x86_64.dmp
   regions-memlist64.dmp
   )

add_unittest_inputs(LLDBMinidumpTests "${test_inputs}")
