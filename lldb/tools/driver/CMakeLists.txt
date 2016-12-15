include(${LLDB_PROJECT_ROOT}/cmake/LLDBDependencies.cmake)

add_lldb_tool(lldb
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
if ((CMAKE_SYSTEM_NAME MATCHES "Windows") OR
    (CMAKE_SYSTEM_NAME MATCHES "NetBSD" ))
  # These targets do not have getopt support, so they rely on the one provided by
  # liblldb. However, getopt is not a part of the liblldb interface, so we have
  # to link against the constituent libraries manually. Note that this is
  # extremely scary as it introduces ODR violations, and it should go away as
  # soon as possible.
  target_link_libraries(lldb ${LLDB_USED_LIBS})
endif()

set_target_properties(lldb PROPERTIES VERSION ${LLDB_VERSION})

