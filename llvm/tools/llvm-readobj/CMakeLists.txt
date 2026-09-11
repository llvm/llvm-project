set(LLVM_LINK_COMPONENTS
  BinaryFormat
  DebugInfoCodeView
  DebugInfoDWARF
  Demangle
  Object
  Option
  Support
  TargetParser
  )

add_llvm_tool(llvm-readobj
  GENERATE_DRIVER
  )

add_subdirectory(lib)
target_link_libraries(llvm-readobj PRIVATE LLVMReadobjDriver)

setup_host_tool(llvm-readobj LLVM_READOBJ llvm_readobj_exe llvm_readobj_target)

add_llvm_tool_symlink(llvm-readelf llvm-readobj)

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(readelf llvm-readobj)
endif()
