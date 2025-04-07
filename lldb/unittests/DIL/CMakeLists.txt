add_lldb_unittest(DILTests
  DILTests.cpp
  DILLexerTests.cpp
  Runner.cpp

  LINK_LIBS
    liblldb
    lldbUtilityHelpers
    lldbValueObject
    LLVMTestingSupport
  )
add_subdirectory(Inputs)
add_dependencies(DILTests test_binary)

add_unittest_inputs(DILTests "test_binary.cpp")
