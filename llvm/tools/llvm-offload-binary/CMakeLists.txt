set(LLVM_LINK_COMPONENTS
  BinaryFormat
  Object
  Support)

add_llvm_tool(llvm-offload-binary
  llvm-offload-binary.cpp

  DEPENDS
  intrinsics_gen
  )
# Legacy binary name to be removed at a later release.
add_llvm_tool_symlink(clang-offload-packager llvm-offload-binary)
