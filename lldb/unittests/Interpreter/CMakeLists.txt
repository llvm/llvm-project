add_lldb_unittest(InterpreterTests
  TestArgs.cpp
  TestCompletion.cpp

  LINK_LIBS
    lldbInterpreter
    lldbUtilityHelpers
  )

target_link_libraries(InterpreterTests
  ${PYTHON_LIBRARY}
  )
