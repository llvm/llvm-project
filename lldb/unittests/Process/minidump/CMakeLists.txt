add_lldb_unittest(LLDBMinidumpTests
  MinidumpParserTest.cpp
  )

set(test_inputs
   linux-x86_64.dmp)

add_unittest_inputs(LLDBMinidumpTests "${test_inputs}")
