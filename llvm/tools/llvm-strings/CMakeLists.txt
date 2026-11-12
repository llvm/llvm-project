set(LLVM_LINK_COMPONENTS
  Core
  Object
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
llvm_tablegen(Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(StringsOptsTableGen)

add_llvm_tool(llvm-strings
  llvm-strings.cpp
  DEPENDS
  StringsOptsTableGen
  )

if(APPLE)
  find_package(Iconv REQUIRED)
  target_link_libraries(llvm-strings PRIVATE Iconv::Iconv)
endif()

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(strings llvm-strings)
endif()
