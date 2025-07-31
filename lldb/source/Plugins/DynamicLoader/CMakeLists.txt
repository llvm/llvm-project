set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND DynamicLoader)
set_property(DIRECTORY PROPERTY LLDB_ACCEPTABLE_PLUGIN_DEPENDENCIES ObjectFile)
set_property(DIRECTORY PROPERTY LLDB_TOLERATED_PLUGIN_DEPENDENCIES
  Process # part of a loop (Process<->DynamicLoader).
  TypeSystem
)

add_subdirectory(Darwin-Kernel)
add_subdirectory(FreeBSD-Kernel)
add_subdirectory(MacOSX-DYLD)
add_subdirectory(POSIX-DYLD)
add_subdirectory(Static)
add_subdirectory(Hexagon-DYLD)
add_subdirectory(Windows-DYLD)
add_subdirectory(wasm-DYLD)
