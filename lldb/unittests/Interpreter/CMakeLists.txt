add_lldb_unittest(InterpreterTests
  TestArgs.cpp

  LINK_LIBS
    lldbInterpreter
  )

target_link_libraries(InterpreterTests
  ${PYTHON_LIBRARY}
  )
