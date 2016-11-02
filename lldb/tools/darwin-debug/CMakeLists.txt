add_lldb_executable(lldb-launcher
  darwin-debug.cpp
  )

install(TARGETS lldb-launcher
  RUNTIME DESTINATION bin)
