if (APPLE AND LLVM_ENABLE_LOCAL_SUBMODULE_VISIBILITY)
  # Work around an issue with the Python headers, which have a modular include
  # inside an extern "C" block.
  remove_module_flags()
endif()

if (LLDB_ENABLE_DYNAMIC_SCRIPTINTERPRETERS)
  set(python_plugin_lib lldbStaticScriptInterpreterPython)
else()
  set(python_plugin_lib lldbPluginScriptInterpreterPython)
endif()

add_lldb_unittest(ScriptInterpreterPythonTests
  PythonDataObjectsTests.cpp
  PythonTestSuite.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbHost
    ${python_plugin_lib}
    LLVMTestingSupport
  )

if(Python3_RPATH)
  set_property(TARGET ScriptInterpreterPythonTests APPEND PROPERTY BUILD_RPATH "${Python3_RPATH}")
endif()
