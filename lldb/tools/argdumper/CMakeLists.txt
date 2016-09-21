add_lldb_executable(lldb-argdumper INCLUDE_IN_FRAMEWORK
  argdumper.cpp
  )

target_link_libraries(lldb-argdumper liblldb)

install(TARGETS lldb-argdumper
  RUNTIME DESTINATION bin)
