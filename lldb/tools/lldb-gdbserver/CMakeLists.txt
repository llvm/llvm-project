set(LLVM_NO_RTTI 1)

if ( CMAKE_SYSTEM_NAME MATCHES "Linux" )
include_directories(
  ../../source/Plugins/Process/Linux
  ../../source/Plugins/Process/POSIX
  )
endif ()

if ( CMAKE_SYSTEM_NAME MATCHES "FreeBSD" )
include_directories(
  ../../source/Plugins/Process/FreeBSD
  ../../source/Plugins/Process/POSIX
  )
endif ()
include_directories(../../source)

include(../../cmake/LLDBDependencies.cmake)

# have to include lldb and lldb-log files since those are not libraries and llgs depends on them
add_lldb_executable(lldb-gdbserver
  lldb-gdbserver.cpp
  ../../source/lldb-log.cpp
  ../../source/lldb.cpp
  )

# The Darwin linker doesn't understand --start-group/--end-group.
if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
  target_link_libraries(lldb-gdbserver
                        -Wl,--start-group ${LLDB_USED_LIBS} -Wl,--end-group)
  target_link_libraries(lldb-gdbserver
                        -Wl,--start-group ${CLANG_USED_LIBS} -Wl,--end-group)
else()
  target_link_libraries(lldb-gdbserver ${LLDB_USED_LIBS})
  target_link_libraries(lldb-gdbserver ${CLANG_USED_LIBS})
endif()
llvm_config(lldb-gdbserver ${LLVM_LINK_COMPONENTS})

target_link_libraries(lldb-gdbserver ${LLDB_SYSTEM_LIBS})

set_target_properties(lldb-gdbserver PROPERTIES VERSION ${LLDB_VERSION})

install(TARGETS lldb-gdbserver
  RUNTIME DESTINATION bin)
