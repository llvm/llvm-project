# FIXME: As we plan to execute llvm-symbolizer binary from compiler-rt
# libraries, it has to be compiled for all supported targets (x86_64, i386 etc).
# This means that we need LLVM libraries to be compiled for these
# targets as well. Currently, there is no support for such a build strategy.

set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  DebugInfoPDB
  Object
  Support
  Symbolize
  )

add_llvm_tool(llvm-symbolizer
  llvm-symbolizer.cpp
  )

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(addr2line llvm-symbolizer)
endif()
