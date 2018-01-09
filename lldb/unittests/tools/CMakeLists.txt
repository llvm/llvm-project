if(CMAKE_SYSTEM_NAME MATCHES "Android|Linux|NetBSD")
  add_subdirectory(lldb-server)
endif()
