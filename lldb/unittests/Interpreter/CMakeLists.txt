add_lldb_unittest(InterpreterTests
  TestCompletion.cpp
  TestOptionArgParser.cpp

  LINK_LIBS
    lldbInterpreter
    lldbUtilityHelpers
  )
