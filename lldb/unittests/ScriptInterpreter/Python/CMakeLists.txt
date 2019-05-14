add_lldb_unittest(ScriptInterpreterPythonTests
  PythonDataObjectsTests.cpp
  PythonExceptionStateTests.cpp
  PythonTestSuite.cpp

  LINK_LIBS
    lldbHost
    lldbPluginScriptInterpreterPython
  LINK_COMPONENTS
    Support
  )