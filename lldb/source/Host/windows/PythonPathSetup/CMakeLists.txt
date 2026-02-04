add_lldb_library(lldbUtilityPythonPathSetup STATIC
  PythonPathSetup.cpp

  LINK_LIBS
    LLVMSupport
)

if(DEFINED LLDB_PYTHON_DLL_RELATIVE_PATH)
  target_compile_definitions(lldbUtilityPythonPathSetup PRIVATE LLDB_PYTHON_DLL_RELATIVE_PATH="${LLDB_PYTHON_DLL_RELATIVE_PATH}")
endif()
if(DEFINED LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME)
  target_compile_definitions(lldbUtilityPythonPathSetup PRIVATE LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME="${LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME}")
endif()
