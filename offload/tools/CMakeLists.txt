set(OFFLOAD_TOOLS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')")
mark_as_advanced(OFFLOAD_TOOLS_INSTALL_DIR)

macro(add_offload_tool name)
  llvm_add_tool(OFFLOAD ${ARGV})
endmacro()

macro(add_offload_tool_symlink name)
  llvm_add_tool_symlink(OFFLOAD ${ARGV})
endmacro()

add_subdirectory(deviceinfo)
add_subdirectory(kernelreplay)
