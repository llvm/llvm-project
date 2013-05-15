add_subdirectory(MacOSX-DYLD)
add_subdirectory(POSIX-DYLD)
add_subdirectory(Static)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  add_subdirectory(Darwin-Kernel)
endif()

