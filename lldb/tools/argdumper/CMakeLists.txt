include(${LLDB_PROJECT_ROOT}/cmake/LLDBDependencies.cmake)

if(NOT LLDB_EXPORT_ALL_SYMBOLS)
  # if liblldb does not export everything we need to directly link these two
  # lldb libraries for lldb_private members
  set(extra_lldb_libs lldbCore lldbUtility)
endif()

add_lldb_tool(lldb-argdumper INCLUDE_IN_FRAMEWORK
  argdumper.cpp

  LINK_LIBS
    liblldb
    ${extra_lldb_libs}
  )

