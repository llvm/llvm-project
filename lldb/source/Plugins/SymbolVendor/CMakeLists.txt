if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  add_subdirectory(MacOSX)
endif()

add_subdirectory(ELF)
add_subdirectory(wasm)
