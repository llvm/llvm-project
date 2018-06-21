add_subdirectory(MacOSX-DYLD)
add_subdirectory(POSIX-DYLD)
add_subdirectory(Static)
add_subdirectory(Hexagon-DYLD)
add_subdirectory(Windows-DYLD)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  add_subdirectory(Darwin-Kernel)
endif()
