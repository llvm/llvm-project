add_lldb_unittest(InterpreterTests
  TestCompletion.cpp
  TestOptionArgParser.cpp
  TestOptionValueFileColonLine.cpp

  LINK_LIBS
    lldbInterpreter
    lldbUtilityHelpers
  )
