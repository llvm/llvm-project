set(LLVM_NO_RTTI 1)
add_lldb_executable(lldb
  Driver.cpp
  #DriverEvents.cpp
  #DriverOptions.cpp
  #DriverPosix.cpp
  IOChannel.cpp
  )

target_link_libraries(lldb liblldb)
# TODO: why isn't this done by add_lldb_executable?
#target_link_libraries(lldb ${LLDB_USED_LIBS})
#llvm_config(lldb ${LLVM_LINK_COMPONENTS})

set_target_properties(lldb PROPERTIES VERSION ${LLDB_VERSION})

install(TARGETS lldb
  RUNTIME DESTINATION bin)
