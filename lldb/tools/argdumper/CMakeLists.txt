include(${LLDB_PROJECT_ROOT}/cmake/LLDBDependencies.cmake)

add_lldb_tool(lldb-argdumper INCLUDE_IN_FRAMEWORK
  argdumper.cpp

  LINK_LIBS
    lldbCore
    lldbUtility
  )

