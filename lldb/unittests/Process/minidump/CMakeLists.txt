add_lldb_unittest(LLDBMinidumpTests
  MinidumpParserTest.cpp
  )

set(test_inputs
   linux-x86_64.dmp
   fizzbuzz_no_heap.dmp)

add_unittest_inputs(LLDBMinidumpTests "${test_inputs}")
