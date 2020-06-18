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

if(PYTHON_RPATH)
  set_property(TARGET ScriptInterpreterPythonTests APPEND PROPERTY BUILD_RPATH "${PYTHON_RPATH}")
endif()