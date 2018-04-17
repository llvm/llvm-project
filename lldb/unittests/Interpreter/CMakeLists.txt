add_lldb_unittest(InterpreterTests
  TestCompletion.cpp
  TestOptionArgParser.cpp

  LINK_LIBS
    lldbInterpreter
    lldbUtilityHelpers
  )

target_link_libraries(InterpreterTests
  PRIVATE
  ${PYTHON_LIBRARY}
  )
