add_lldb_executable(darwin-debug INCLUDE_IN_FRAMEWORK
  darwin-debug.cpp
  )

install(TARGETS darwin-debug
  RUNTIME DESTINATION bin)
