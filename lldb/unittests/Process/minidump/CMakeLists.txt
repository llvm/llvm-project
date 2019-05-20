add_lldb_unittest(LLDBMinidumpTests
  MinidumpParserTest.cpp
  RegisterContextMinidumpTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbTarget
    lldbPluginProcessUtility
    lldbPluginProcessMinidump
    lldbUtilityHelpers
    LLVMTestingSupport
  LINK_COMPONENTS
    ObjectYAML
    Support
  )

set(test_inputs
   fizzbuzz_no_heap.dmp
   fizzbuzz_wow64.dmp
   linux-x86_64.dmp
   linux-x86_64_not_crashed.dmp
   regions-memlist64.dmp
   )

add_unittest_inputs(LLDBMinidumpTests "${test_inputs}")
