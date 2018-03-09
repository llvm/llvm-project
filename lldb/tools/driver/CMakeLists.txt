if ((CMAKE_SYSTEM_NAME MATCHES "Windows") OR
    (CMAKE_SYSTEM_NAME MATCHES "NetBSD" ))
  # These targets do not have getopt support, so they rely on the one provided by
  # liblldb. However, getopt is not a part of the liblldb interface, so we have
  # to link against the constituent libraries manually. Note that this is
  # extremely scary as it introduces ODR violations, and it should go away as
  # soon as possible.
  set(host_lib lldbHost)
endif()

add_lldb_tool(lldb
  Driver.cpp
  Platform.cpp

  LINK_LIBS
    liblldb
    ${host_lib}

  LINK_COMPONENTS
    Support
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
