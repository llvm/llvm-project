add_lldb_unittest(ScriptInterpreterPythonTests
  PythonDataObjectsTests.cpp
  PythonExceptionStateTests.cpp
  PythonTestSuite.cpp

  LINK_LIBS
    lldbHost
    lldbPluginScriptInterpreterPython
    ${PYTHON_LIBRARY}
    ${lldb_python_wrapper}
  LINK_COMPONENTS
    Support
  )
