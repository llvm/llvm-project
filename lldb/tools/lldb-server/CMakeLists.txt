if ( CMAKE_SYSTEM_NAME MATCHES "Linux" )
include_directories(
  ../../../../llvm/include
  ../../source/Plugins/Process/Linux
  ../../source/Plugins/Process/POSIX
  )
endif ()

if ( CMAKE_SYSTEM_NAME MATCHES "FreeBSD" )
include_directories(
  ../../../../llvm/include
  ../../source/Plugins/Process/FreeBSD
  ../../source/Plugins/Process/POSIX
  )
endif ()

if ( CMAKE_SYSTEM_NAME MATCHES "NetBSD" )
include_directories(
  ../../../../llvm/include
  ../../source/Plugins/Process/POSIX
  )
endif ()

include_directories(../../source)

include(../../cmake/LLDBDependencies.cmake)

add_lldb_executable(lldb-server INCLUDE_IN_FRAMEWORK
    Acceptor.cpp
    lldb-gdbserver.cpp
    lldb-platform.cpp
    lldb-server.cpp
    LLDBServerUtilities.cpp
)

# The Darwin linker doesn't understand --start-group/--end-group.
if (LLDB_LINKER_SUPPORTS_GROUPS)
  target_link_libraries(lldb-server
                        -Wl,--start-group ${LLDB_USED_LIBS} -Wl,--end-group)
  target_link_libraries(lldb-server
                        -Wl,--start-group ${CLANG_USED_LIBS} -Wl,--end-group)
else()
  target_link_libraries(lldb-server ${LLDB_USED_LIBS})
  target_link_libraries(lldb-server ${CLANG_USED_LIBS})
endif()
llvm_config(lldb-server ${LLVM_LINK_COMPONENTS})

target_link_libraries(lldb-server ${LLDB_SYSTEM_LIBS})

set_target_properties(lldb-server PROPERTIES VERSION ${LLDB_VERSION})

install(TARGETS lldb-server
  RUNTIME DESTINATION bin)
