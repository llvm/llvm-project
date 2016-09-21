add_lldb_executable(lldb-launcher INCLUDE_IN_FRAMEWORK
  darwin-debug.cpp
  )

install(TARGETS lldb-launcher
  RUNTIME DESTINATION bin)
