add_lldb_unittest(InterpreterTests
  TestCompletion.cpp
  TestOptionArgParser.cpp
  TestOptionValue.cpp
  TestOptionValueFileColonLine.cpp

  LINK_LIBS
    lldbInterpreter
    lldbUtilityHelpers
  )
