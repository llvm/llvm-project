set(LLVM_LINK_COMPONENTS
  Demangle
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(CxxfiltOptsTableGen)

add_llvm_tool(llvm-cxxfilt
  llvm-cxxfilt.cpp

  DEPENDS
  CxxfiltOptsTableGen
  )

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(c++filt llvm-cxxfilt)
endif()
