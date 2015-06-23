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

add_lldb_executable(lldb-platform
  lldb-platform.cpp
  ../../source/lldb-log.cpp
  ../../source/lldb.cpp
  ${LLDB_WRAP_PYTHON}
  ${LLDB_VERS_GENERATED_FILE}
  )

if (LLDB_WRAP_PYTHON OR LLDB_VERS_GENERATED_FILE)
  add_dependencies(lldb-platform swig_wrapper)
endif()

# The Darwin linker doesn't understand --start-group/--end-group.
if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
  target_link_libraries(lldb-platform
                        -Wl,--start-group ${LLDB_USED_LIBS} -Wl,--end-group)
  target_link_libraries(lldb-platform
                        -Wl,--start-group ${CLANG_USED_LIBS} -Wl,--end-group)
else()
  target_link_libraries(lldb-platform ${LLDB_USED_LIBS})
  target_link_libraries(lldb-platform ${CLANG_USED_LIBS})
endif()
llvm_config(lldb-platform ${LLVM_LINK_COMPONENTS})

target_link_libraries(lldb-platform ${LLDB_SYSTEM_LIBS})

set_target_properties(lldb-platform PROPERTIES VERSION ${LLDB_VERSION})

install(TARGETS lldb-platform
  RUNTIME DESTINATION bin)
