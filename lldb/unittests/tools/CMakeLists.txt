add_subdirectory(lldb-mi)
if(CMAKE_SYSTEM_NAME MATCHES "Android|Darwin|Linux|NetBSD")
  if ((CMAKE_SYSTEM_NAME MATCHES "Darwin" AND SKIP_TEST_DEBUGSERVER) OR (NOT CMAKE_SYSTEM_NAME MATCHES "Darwin" AND SKIP_LLDB_SERVER_BUILD))
    # These tests are meant to test lldb-server/debugserver in isolation, and
    # don't provide any value if run against a server copied from somewhere.
  else()
    add_subdirectory(lldb-server)
  endif()
endif()
