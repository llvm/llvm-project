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
  LINK_COMPONENTS
    Support
  )

set(test_inputs
   bad_duplicate_streams.dmp
   bad_overlapping_streams.dmp
   fizzbuzz_no_heap.dmp
   fizzbuzz_wow64.dmp
   linux-i386.dmp
   linux-x86_64.dmp
   linux-x86_64_not_crashed.dmp
   memory-list-not-padded.dmp
   memory-list-padded.dmp
   module-list-not-padded.dmp
   module-list-padded.dmp
   modules-dup-min-addr.dmp
   modules-order.dmp
   regions-linux-map.dmp
   regions-memlist.dmp
   regions-memlist64.dmp
   thread-list-not-padded.dmp
   thread-list-padded.dmp
   )

add_unittest_inputs(LLDBMinidumpTests "${test_inputs}")
