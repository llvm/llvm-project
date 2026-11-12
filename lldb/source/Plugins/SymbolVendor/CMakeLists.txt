set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND SymbolVendor)
set_property(DIRECTORY PROPERTY LLDB_ACCEPTABLE_PLUGIN_DEPENDENCIES ObjectFile)

add_subdirectory(ELF)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  add_subdirectory(MacOSX)
endif()

add_subdirectory(PECOFF)
add_subdirectory(wasm)
