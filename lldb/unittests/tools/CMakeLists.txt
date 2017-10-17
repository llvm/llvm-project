if(CMAKE_SYSTEM_NAME MATCHES "Android|Darwin|Linux|NetBSD")
  add_subdirectory(lldb-server)
endif()
