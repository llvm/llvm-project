add_lldb_unittest(ScriptInterpreterPythonTests
  PythonDataObjectsTests.cpp
  PythonExceptionStateTests.cpp
  PythonTestSuite.cpp
  )

  target_link_libraries(ScriptInterpreterPythonTests lldbPluginScriptInterpreterPython ${PYTHON_LIBRARY})
  