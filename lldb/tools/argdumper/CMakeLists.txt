add_lldb_tool(lldb-argdumper
  argdumper.cpp

  LINK_LIBS
    lldbUtility
  )

if(LLDB_BUILD_FRAMEWORK)
  lldb_add_to_framework(lldb-argdumper)
endif()
