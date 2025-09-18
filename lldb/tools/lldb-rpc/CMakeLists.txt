include(CheckCXXCompilerFlag)
# Umbrella target for the entire framework is a default target.
add_custom_target(lldb-rpc ALL)

if(LLDB_CODESIGN_IDENTITY)
  # Use explicit LLDB identity
  set(LLVM_CODESIGNING_IDENTITY ${LLDB_CODESIGN_IDENTITY})
else()
  # Use explicit LLVM identity or default to ad-hoc signing if empty
  if(NOT LLVM_CODESIGNING_IDENTITY)
    set(LLVM_CODESIGNING_IDENTITY -)
  endif()
endif()

# LLDBRPCGeneration.cmake needs the LLDB_RPC_GEN_EXE variable
# which gets defined in the lldb-rpc-gen folder, so we're adding
# this folder before we add that file.
add_lldb_tool_subdirectory(lldb-rpc-gen)
include(${CMAKE_CURRENT_SOURCE_DIR}/LLDBRPCGeneration.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/LLDBRPCHeaders.cmake)

add_dependencies(lldb-rpc lldb-rpc-generate-sources liblldbrpc-headers)
