add_lldb_unittest(ScriptInterpreterPythonTests
  PythonDataObjectsTests.cpp
  PythonTestSuite.cpp

  LINK_LIBS
    lldbHost
    lldbPluginScriptInterpreterPython
    LLVMTestingSupport
  LINK_COMPONENTS
    Support
  )