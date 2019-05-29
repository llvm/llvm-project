add_lldb_tool(darwin-debug
  darwin-debug.cpp
  )

if(LLDB_BUILD_FRAMEWORK)
  lldb_add_to_framework(darwin-debug)
endif()
