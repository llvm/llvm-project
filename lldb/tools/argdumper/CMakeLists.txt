add_lldb_executable(lldb-argdumper
  argdumper.cpp
  )

target_link_libraries(lldb-argdumper liblldb)

install(TARGETS lldb-argdumper
  RUNTIME DESTINATION bin)
