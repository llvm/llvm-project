add_lldb_executable(lldb
  Driver.cpp
  Platform.cpp
  )

if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  add_definitions( -DIMPORT_LIBLLDB )
endif()

# Add lldb dependency on lldb-server if we can use it.
if ( LLDB_CAN_USE_LLDB_SERVER )
  add_dependencies(lldb lldb-server)
endif()

# Add lldb dependency on debugserver if we can use it.
if ( LLDB_CAN_USE_DEBUGSERVER )
  add_dependencies(lldb debugserver)
endif()

target_link_libraries(lldb liblldb)
# TODO: why isn't this done by add_lldb_executable?
#target_link_libraries(lldb ${LLDB_USED_LIBS})
#llvm_config(lldb ${LLVM_LINK_COMPONENTS})

set_target_properties(lldb PROPERTIES VERSION ${LLDB_VERSION})

install(TARGETS lldb
  RUNTIME DESTINATION bin)
