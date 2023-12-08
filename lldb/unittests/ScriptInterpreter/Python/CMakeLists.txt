if (APPLE AND LLVM_ENABLE_LOCAL_SUBMODULE_VISIBILITY)
  # Work around an issue with the Python headers, which have a modular include
  # inside an extern "C" block.
  remove_module_flags()
endif()

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

if(Python3_RPATH)
  set_property(TARGET ScriptInterpreterPythonTests APPEND PROPERTY BUILD_RPATH "${Python3_RPATH}")
endif()
